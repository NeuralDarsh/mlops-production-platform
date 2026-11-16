# Distributed Systems & MLOps Control Plane: Term-based leader election, majority voting, and log replication

import json
import random
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class NodeRole(Enum):
    FOLLOWER = "FOLLOWER"
    CANDIDATE = "CANDIDATE"
    LEADER = "LEADER"


class LogEntry:
    """Represents a single replicated command appended to the Raft consensus log."""
    def __init__(self, term: int, index: int, command: Dict[str, Any]):
        self.term = term
        self.index = index
        self.command = command

    def to_dict(self) -> Dict[str, Any]:
        return {"term": self.term, "index": self.index, "command": self.command}


class RaftConsensusNode:
    """
    Implements core Raft consensus mechanics: randomized election timers,
    term-based majority voting, and replicated state-machine log commitment.
    """
    def __init__(self, node_id: str, peer_ids: List[str]):
        self.node_id = node_id
        self.peer_ids = peer_ids
        self.is_online = True
        self.current_term = 0
        self.voted_for: Optional[str] = None
        self.log: List[LogEntry] = []
        self.role = NodeRole.FOLLOWER
        self.commit_index = -1
        self.state_machine: Dict[str, Any] = {}

        self.election_timeout = random.randint(3, 6)
        self.ticks_since_heartbeat = 0

    def _step_down(self, new_term: int):
        """Reverts node to FOLLOWER whenever a higher term is discovered."""
        self.current_term = new_term
        self.role = NodeRole.FOLLOWER
        self.voted_for = None
        self.ticks_since_heartbeat = 0

    def handle_request_vote(self, candidate_id: str, term: int, last_log_index: int, last_log_term: int) -> Tuple[int, bool]:
        """Evaluates incoming RequestVote RPC according to Raft log-completeness rules."""
        if not self.is_online:
            return self.current_term, False

        if term > self.current_term:
            self._step_down(term)

        if term < self.current_term:
            return self.current_term, False

        my_last_index = len(self.log) - 1
        my_last_term = self.log[my_last_index].term if my_last_index >= 0 else 0
        log_is_up_to_date = (last_log_term > my_last_term) or (
            last_log_term == my_last_term and last_log_index >= my_last_index
        )

        if (self.voted_for is None or self.voted_for == candidate_id) and log_is_up_to_date:
            self.voted_for = candidate_id
            self.ticks_since_heartbeat = 0
            return self.current_term, True

        return self.current_term, False

    def handle_append_entries(
        self,
        leader_id: str,
        term: int,
        entries: List[LogEntry],
        leader_commit: int
    ) -> Tuple[int, bool]:
        """Handles heartbeat and log replication RPCs from the active leader."""
        if not self.is_online:
            return self.current_term, False

        if term < self.current_term:
            return self.current_term, False

        if term > self.current_term or self.role != NodeRole.FOLLOWER:
            self._step_down(term)

        self.ticks_since_heartbeat = 0
        for entry in entries:
            if entry.index >= len(self.log):
                self.log.append(LogEntry(entry.term, entry.index, dict(entry.command)))

        if leader_commit > self.commit_index:
            self.commit_index = min(leader_commit, len(self.log) - 1)
            self._apply_committed_entries()

        return self.current_term, True

    def _apply_committed_entries(self):
        """Applies all committed log commands deterministically to local key-value state."""
        for idx in range(self.commit_index + 1):
            cmd = self.log[idx].command
            if cmd.get("op") == "SET_ACTIVE_MODEL":
                self.state_machine[cmd["pipeline"]] = cmd["model_version"]

    def start_election(self, cluster: Dict[str, "RaftConsensusNode"]):
        """Transitions to CANDIDATE, increments term, and solicits votes from peers."""
        self.current_term += 1
        self.role = NodeRole.CANDIDATE
        self.voted_for = self.node_id
        self.ticks_since_heartbeat = 0

        votes_received = 1 
        last_idx = len(self.log) - 1
        last_term = self.log[last_idx].term if last_idx >= 0 else 0

        print(f"[{self.node_id}] Election timeout! Starting election for Term {self.current_term}...")

        for peer_id in self.peer_ids:
            peer = cluster.get(peer_id)
            if peer:
                resp_term, granted = peer.handle_request_vote(
                    self.node_id, self.current_term, last_idx, last_term
                )
                if resp_term > self.current_term:
                    self._step_down(resp_term)
                    return
                if granted:
                    votes_received += 1

        quorum = (len(self.peer_ids) + 1) // 2 + 1
        if self.role == NodeRole.CANDIDATE and votes_received >= quorum:
            self.role = NodeRole.LEADER
            print(f"[{self.node_id}] Won election with {votes_received}/{len(self.peer_ids) + 1} votes! Became LEADER (Term {self.current_term}).")
            self.broadcast_heartbeat_and_replicate(cluster)

    def propose_command(self, command: Dict[str, Any], cluster: Dict[str, "RaftConsensusNode"]) -> bool:
        """Client entry point: appends command on leader and replicates to majority."""
        if self.role != NodeRole.LEADER or not self.is_online:
            return False

        new_entry = LogEntry(term=self.current_term, index=len(self.log), command=command)
        self.log.append(new_entry)
        print(f"[LEADER {self.node_id}] Appended entry #{new_entry.index}: {command}")
        self.broadcast_heartbeat_and_replicate(cluster)
        return True

    def broadcast_heartbeat_and_replicate(self, cluster: Dict[str, "RaftConsensusNode"]):
        """Replicates log entries to followers and commits once majority acknowledges."""
        if self.role != NodeRole.LEADER or not self.is_online:
            return

        acks = 1 
        for peer_id in self.peer_ids:
            peer = cluster.get(peer_id)
            if peer:
                resp_term, success = peer.handle_append_entries(
                    self.node_id, self.current_term, self.log, self.commit_index
                )
                if resp_term > self.current_term:
                    self._step_down(resp_term)
                    return
                if success:
                    acks += 1

        quorum = (len(self.peer_ids) + 1) // 2 + 1
        if acks >= quorum and len(self.log) - 1 > self.commit_index:
            self.commit_index = len(self.log) - 1
            self._apply_committed_entries()
            print(f"[QUORUM COMMIT] Index #{self.commit_index} committed across {acks}/{len(self.peer_ids) + 1} nodes.")
            for peer_id in self.peer_ids:
                peer = cluster.get(peer_id)
                if peer:
                    peer.handle_append_entries(self.node_id, self.current_term, self.log, self.commit_index)

    def tick(self, cluster: Dict[str, "RaftConsensusNode"]):
        """Advances node logical clock by one tick."""
        if not self.is_online:
            return

        if self.role == NodeRole.LEADER:
            self.broadcast_heartbeat_and_replicate(cluster)
        else:
            self.ticks_since_heartbeat += 1
            if self.ticks_since_heartbeat >= self.election_timeout:
                self.start_election(cluster)


if __name__ == "__main__":
    print("--- MLOps Control Plane: Raft Leader Election & Consensus Engine ---\n")
    random.seed(7)

    node_ids = ["ctrl_node_1", "ctrl_node_2", "ctrl_node_3", "ctrl_node_4", "ctrl_node_5"]
    cluster = {
        nid: RaftConsensusNode(nid, [p for p in node_ids if p != nid])
        for nid in node_ids
    }

    print("Step 1: Running cluster ticks to elect initial control-plane LEADER:")
    leader: Optional[RaftConsensusNode] = None
    for _ in range(10):
        for nid in node_ids:
            cluster[nid].tick(cluster)
            if cluster[nid].role == NodeRole.LEADER:
                leader = cluster[nid]
                break
        if leader:
            break
    print("\nStep 2: Replicating model promotion commands across consensus log:")
    if leader:
        leader.propose_command({"op": "SET_ACTIVE_MODEL", "pipeline": "fraud_scorer", "model_version": "v3.2.0"}, cluster)
        leader.propose_command({"op": "SET_ACTIVE_MODEL", "pipeline": "recsys_ranker", "model_version": "v1.9.4"}, cluster)

    print(f"\nStep 3: Simulating crash of active LEADER '{leader.node_id}' -> Triggering failover election:")
    if leader:
        leader.is_online = False

    new_leader: Optional[RaftConsensusNode] = None
    for _ in range(12):
        for nid in node_ids:
            node = cluster[nid]
            if node.is_online:
                node.tick(cluster)
                if node.role == NodeRole.LEADER:
                    new_leader = node
                    break
        if new_leader:
            break
    if new_leader:
        print(f"\nStep 4: Committing post-failover promotion via new LEADER '{new_leader.node_id}':")
        new_leader.propose_command({"op": "SET_ACTIVE_MODEL", "pipeline": "fraud_scorer", "model_version": "v3.3.0-hotfix"}, cluster)

    print("\nReplicated Control-Plane State Machine Audit (Across Surviving Nodes):")
    audit = {
        nid: {
            "role": node.role.value,
            "term": node.current_term,
            "commit_index": node.commit_index,
            "state_machine": node.state_machine
        }
        for nid, node in cluster.items() if node.is_online
    }
    print(json.dumps(audit, indent=2))