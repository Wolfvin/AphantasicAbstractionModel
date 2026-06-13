//! # v1.0.0 Cognitive Scenario Tests
//!
//! These are NOT unit tests. They are **cognitive scenarios** that prove the system
//! actually works as claimed — that it can detect contradictions, reason about hidden
//! meaning, accumulate confidence over time, ask the right questions, and discover
//! structural equivalence without co-occurrence.

#![allow(clippy::field_reassign_with_default)]

mod helpers;

#[cfg(test)]
mod core_priority;

#[cfg(test)]
mod bonus;

#[cfg(test)]
mod blind_spot;

#[cfg(test)]
mod integration;

#[cfg(test)]
mod condition_consequence;

#[cfg(test)]
mod verbalize;

#[cfg(test)]
mod phase_jp;

#[cfg(test)]
mod audit;

#[cfg(test)]
mod graph_query;
