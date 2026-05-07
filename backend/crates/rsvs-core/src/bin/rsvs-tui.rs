//! RSVS TUI v9.0 — Split-View with Confidence Heatmap + Relation Panel
//!
//! Layout (Option C):
//! ┌─ ATOMS ─────────────────┐┌─ SELECTED: dokter ──────────┐
//! │ ●● dokter      [0.82] ██││ Layer: 1                    │
//! │ ●● petani      [0.71] █▓││ Tier:  Stable               │
//! │ ○  gunung      [0.65] █▓││ Senses: 2                   │
//! │ ○  laut        [0.44] █░││ Compositions:               │
//! │ ·  sejarah     [0.21] ░ ││  → agent    (seed)          │
//! │ ·  padi        [0.18] ░ ││  → rumah    (0.71)          │
//! │                         ││ Related:                    │
//! │ [↑↓] navigate [Enter]   ││  pasien(0.88) perawat(0.72) │
//! └─────────────────────────┘└─────────────────────────────┘
//!
//! ┌─ OUTPUT ──────────────────────────────────────────────────┐
//! │ (command output from ingest, appraise, etc.)              │
//! └───────────────────────────────────────────────────────────┘
//! ┌ NORMAL > _                                               ┐
//! │ [I]ngest [A]ppraise [C]ontext [R]elate [Q]uit [?]help   │

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame, Terminal,
};
use rsvs::{
    AppraiseVerdict, IngestStats, PipelineConfig, RelateResult, Rsvs,
};
use rsvs::session::SessionGraph;
use std::io;

// ---------------------------------------------------------------------------
// Atom info for the left panel
// ---------------------------------------------------------------------------

/// Simplified atom info for display in the ATOMS panel.
#[derive(Debug, Clone)]
struct AtomInfo {
    label: String,
    confidence: f32,
    tier_str: String,      // "Stable", "Candidate", "New", "Deprecated", "Quarantine"
    layer: u32,
    sense_count: usize,
    is_seed: bool,
    #[allow(dead_code)]
    node_id: u32,
    compositions: Vec<(String, f32, bool)>, // (label, confidence, is_seed)
    related: Vec<(String, f32)>,            // (label, score)
    coherence: Option<f32>,
    grounding_score: Option<f32>,
}

fn snapshot_to_atoms(rsvs: &Rsvs) -> Vec<AtomInfo> {
    let snapshot = rsvs.snapshot_v1();
    let mut atoms: Vec<AtomInfo> = Vec::new();

    for node in &snapshot.nodes {
        // Resolve composition labels
        let compositions: Vec<(String, f32, bool)> = node
            .compositions
            .iter()
            .filter_map(|comp| {
                rsvs.graph.get_node(comp.node_id).map(|n| {
                    let conf = rsvs.autonomy.confidence(n.id).unwrap_or(n.confidence);
                    (n.label.clone(), conf, n.is_seed)
                })
            })
            .collect();

        // Get related nodes via relate()
        let related: Vec<(String, f32)> = if let Some(result) = rsvs.relate(&node.label) {
            result
                .structural_relations
                .iter()
                .take(8)
                .filter_map(|(id, score)| {
                    rsvs.graph.get_node(*id).map(|n| (n.label.clone(), *score))
                })
                .collect()
        } else {
            Vec::new()
        };

        atoms.push(AtomInfo {
            label: node.label.clone(),
            confidence: node.confidence,
            tier_str: node.status.clone(),
            layer: node.layer,
            sense_count: node.sense_count,
            is_seed: node.is_seed,
            node_id: node.id,
            compositions,
            related,
            coherence: node.coherence,
            grounding_score: node.grounding_score,
        });
    }

    // Sort by confidence descending, then label ascending
    atoms.sort_by(|a, b| {
        b.confidence
            .total_cmp(&a.confidence)
            .then_with(|| a.label.cmp(&b.label))
    });

    atoms
}

// ---------------------------------------------------------------------------
// App state
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum Mode {
    Normal,
    Insert,
    Appraise,
    ContextStep1,
    ContextStep2,
    Relate,
    Help,
}

impl std::fmt::Display for Mode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Mode::Normal => write!(f, "NORMAL"),
            Mode::Insert => write!(f, "INSERT"),
            Mode::Appraise => write!(f, "APPRAISE"),
            Mode::ContextStep1 => write!(f, "CONTEXT(1)"),
            Mode::ContextStep2 => write!(f, "CONTEXT(2)"),
            Mode::Relate => write!(f, "RELATE"),
            Mode::Help => write!(f, "HELP"),
        }
    }
}

struct App {
    rsvs: Rsvs,
    mode: Mode,
    input: String,
    output: Vec<Line<'static>>,
    context_buffer: String,
    // Atom browser state
    atoms: Vec<AtomInfo>,
    selected_atom: usize,
    #[allow(dead_code)]
    atom_scroll: u16,
    output_scroll: u16,
}

impl App {
    fn new() -> Self {
        let config = PipelineConfig {
            entity_promote_n: 3,
            ..PipelineConfig::default()
        };
        let rsvs = Rsvs::new(config).expect("Failed to initialize RSVS");
        let atoms = snapshot_to_atoms(&rsvs);
        Self {
            rsvs,
            mode: Mode::Normal,
            input: String::new(),
            output: vec![Line::from(Span::styled(
                "RSVS v9.0 TUI ready. Press ? for help. ↑↓ to browse atoms.",
                Style::default().fg(Color::Cyan),
            ))],
            context_buffer: String::new(),
            atoms,
            selected_atom: 0,
            atom_scroll: 0,
            output_scroll: 0,
        }
    }

    fn refresh_atoms(&mut self) {
        self.atoms = snapshot_to_atoms(&self.rsvs);
        if self.selected_atom >= self.atoms.len() && !self.atoms.is_empty() {
            self.selected_atom = self.atoms.len() - 1;
        }
    }

    fn push_output(&mut self, lines: Vec<Line<'static>>) {
        self.output.extend(lines);
        self.output_scroll = 0;
    }

    fn process_input(&mut self, input: String) {
        match self.mode {
            Mode::Insert => {
                if input.is_empty() {
                    self.push_output(vec![Line::from(Span::styled(
                        "Empty input — nothing ingested.",
                        Style::default().fg(Color::Yellow),
                    ))]);
                } else {
                    match self.rsvs.ingest_text(&input) {
                        Ok(stats) => {
                            self.push_output(format_ingest_result(&stats));
                            self.refresh_atoms();
                        }
                        Err(e) => {
                            self.push_output(vec![Line::from(Span::styled(
                                format!("Error: {}", e),
                                Style::default().fg(Color::Red),
                            ))]);
                        }
                    }
                }
                self.mode = Mode::Normal;
            }
            Mode::Appraise => {
                if input.is_empty() {
                    self.push_output(vec![Line::from(Span::styled(
                        "Empty statement — nothing appraised.",
                        Style::default().fg(Color::Yellow),
                    ))]);
                } else {
                    let verdict = self.rsvs.appraise_verbose(&input);
                    self.push_output(format_verdict_result(&verdict, "Appraise"));
                }
                self.mode = Mode::Normal;
            }
            Mode::ContextStep1 => {
                if input.is_empty() {
                    self.push_output(vec![Line::from(Span::styled(
                        "Empty context — please provide a story/context.",
                        Style::default().fg(Color::Yellow),
                    ))]);
                    self.mode = Mode::Normal;
                } else {
                    self.context_buffer = input;
                    self.push_output(vec![Line::from(Span::styled(
                        "Context saved. Now enter the statement to test:",
                        Style::default().fg(Color::Cyan),
                    ))]);
                    self.mode = Mode::ContextStep2;
                }
            }
            Mode::ContextStep2 => {
                if input.is_empty() {
                    self.push_output(vec![Line::from(Span::styled(
                        "Empty statement — nothing appraised.",
                        Style::default().fg(Color::Yellow),
                    ))]);
                } else {
                    match SessionGraph::new(&self.context_buffer, self.rsvs.config.clone()) {
                        Ok(session) => {
                            let sstats = session.stats();
                            self.push_output(vec![Line::from(Span::styled(
                                format!(
                                    "[Session] {} sentences → {} atoms induced",
                                    sstats.sentences_ingested, sstats.atoms_induced
                                ),
                                Style::default().fg(Color::DarkGray),
                            ))]);
                            let ctx_preview = if self.context_buffer.len() > 60 {
                                format!("[Context] {}...", &self.context_buffer[..60])
                            } else {
                                format!("[Context] {}", self.context_buffer)
                            };
                            self.push_output(vec![Line::from(Span::styled(
                                ctx_preview,
                                Style::default().fg(Color::DarkGray),
                            ))]);
                            self.push_output(vec![Line::from(Span::raw(
                                "─".repeat(60),
                            ))]);
                            let verdict = session.appraise(&input);
                            self.push_output(format_verdict_result(
                                &verdict,
                                "Contextual Appraise",
                            ));
                            self.push_output(vec![Line::from(Span::styled(
                                "Note: Main graph was NOT modified.",
                                Style::default().fg(Color::DarkGray),
                            ))]);
                        }
                        Err(e) => {
                            self.push_output(vec![Line::from(Span::styled(
                                format!("Session error: {}", e),
                                Style::default().fg(Color::Red),
                            ))]);
                        }
                    }
                }
                self.context_buffer.clear();
                self.mode = Mode::Normal;
            }
            Mode::Relate => {
                if input.is_empty() {
                    self.push_output(vec![Line::from(Span::styled(
                        "Empty concept — nothing related.",
                        Style::default().fg(Color::Yellow),
                    ))]);
                } else {
                    match self.rsvs.relate(&input) {
                        Some(result) => {
                            self.push_output(format_relate_result(&result, &self.rsvs));
                        }
                        None => {
                            self.push_output(vec![Line::from(Span::styled(
                                format!("Concept '{}' not found in graph.", input),
                                Style::default().fg(Color::Red),
                            ))]);
                        }
                    }
                }
                self.mode = Mode::Normal;
            }
            _ => {}
        }
    }

    fn handle_normal_key(&mut self, key: KeyCode) {
        match key {
            KeyCode::Char('i') => {
                self.mode = Mode::Insert;
                self.push_output(vec![Line::from(Span::styled(
                    "INSERT mode — type text to ingest, Enter to submit, Esc to cancel:",
                    Style::default().fg(Color::Green),
                ))]);
            }
            KeyCode::Char('a') => {
                self.mode = Mode::Appraise;
                self.push_output(vec![Line::from(Span::styled(
                    "APPRAISE mode — type statement to evaluate:",
                    Style::default().fg(Color::Yellow),
                ))]);
            }
            KeyCode::Char('c') => {
                self.mode = Mode::ContextStep1;
                self.context_buffer.clear();
                self.push_output(vec![Line::from(Span::styled(
                    "CONTEXT mode — Step 1: enter story/context:",
                    Style::default().fg(Color::Magenta),
                ))]);
            }
            KeyCode::Char('r') => {
                self.mode = Mode::Relate;
                self.push_output(vec![Line::from(Span::styled(
                    "RELATE mode — type concept to find relations:",
                    Style::default().fg(Color::Blue),
                ))]);
            }
            KeyCode::Char('?') => {
                if self.mode == Mode::Help {
                    self.mode = Mode::Normal;
                } else {
                    self.mode = Mode::Help;
                }
            }
            KeyCode::Up => {
                if !self.atoms.is_empty() && self.selected_atom > 0 {
                    self.selected_atom -= 1;
                }
            }
            KeyCode::Down => {
                if !self.atoms.is_empty() && self.selected_atom < self.atoms.len() - 1 {
                    self.selected_atom += 1;
                }
            }
            KeyCode::Enter => {
                // Select atom — show relate in output
                if let Some(atom) = self.atoms.get(self.selected_atom) {
                    if let Some(result) = self.rsvs.relate(&atom.label) {
                        self.push_output(format_relate_result(&result, &self.rsvs));
                    }
                }
            }
            KeyCode::Char('q') => {}
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Confidence bar renderer
// ---------------------------------------------------------------------------

/// Map confidence (0.0–1.0) to a visual bar string using block characters.
/// 8 characters wide: ██ = full, ▓ = three-quarters, ░ = quarter, space = empty
fn confidence_bar(confidence: f32, width: usize) -> String {
    let total_units = width * 8; // each char has 8 sub-units
    let filled = (confidence * total_units as f32).round() as usize;
    let mut bar = String::new();
    let mut remaining = filled.min(total_units);

    for _ in 0..width {
        if remaining >= 8 {
            bar.push('\u{2588}'); // █ full block
            remaining -= 8;
        } else if remaining >= 6 {
            bar.push('\u{2593}'); // ▓ three-quarters
            remaining = 0;
        } else if remaining >= 4 {
            bar.push('\u{2592}'); // ▒ half
            remaining = 0;
        } else if remaining >= 2 {
            bar.push('\u{2591}'); // ░ quarter
            remaining = 0;
        } else {
            bar.push(' ');
        }
    }
    bar
}

/// Tier bullet: ●● = Stable, ● = Candidate, ○ = New, · = Deprecated/Quarantine
fn tier_bullet(status: &str) -> (&'static str, Color) {
    match status {
        "stable" => ("\u{25cf}\u{25cf}", Color::Green),   // ●●
        "candidate" => ("\u{25cf}", Color::Yellow),        // ●
        "new" => ("\u{25cb}", Color::Cyan),                // ○
        "deprecated" => ("\u{00b7}", Color::DarkGray),     // ·
        "quarantine" => ("\u{00b7}", Color::Red),          // ·
        _ => (" ", Color::White),
    }
}

fn confidence_color(conf: f32) -> Color {
    if conf >= 0.7 {
        Color::Green
    } else if conf >= 0.4 {
        Color::Yellow
    } else if conf >= 0.2 {
        Color::Cyan
    } else {
        Color::DarkGray
    }
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

fn verdict_color(verdict: &str) -> Color {
    match verdict {
        "consistent" => Color::Green,
        "partial" => Color::Yellow,
        "novel" => Color::Cyan,
        "clash" => Color::Red,
        "mixed" => Color::Magenta,
        "disagree" => Color::Red,
        _ => Color::White,
    }
}

fn score_color(score: f32) -> Color {
    if score > 0.7 {
        Color::Green
    } else if score > 0.4 {
        Color::Yellow
    } else {
        Color::Red
    }
}

fn format_ingest_result(stats: &IngestStats) -> Vec<Line<'static>> {
    vec![
        Line::from(Span::styled(
            "Ingest complete",
            Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
        )),
        Line::from(format!(
            "  sentences: {} | atoms: {} | senses: {}/{} | compositions: {}",
            stats.sentences_processed,
            stats.atoms_promoted,
            stats.sense_created,
            stats.sense_assigned,
            stats.compositions_induced,
        )),
        Line::from(Span::raw("─".repeat(60))),
    ]
}

fn format_verdict_result(verdict: &AppraiseVerdict, label: &str) -> Vec<Line<'static>> {
    let mut lines = Vec::new();

    let confidence_label = if verdict.confidence_gap > 30.0 {
        " [CONFIDENT]"
    } else if verdict.confidence_gap > 10.0 {
        " [MODERATE]"
    } else if verdict.confidence_gap > 0.0 {
        " [AMBIGUOUS]"
    } else {
        " [INVERTED]"
    };
    let ctx_label = if verdict.is_contextual {
        " [CONTEXTUAL]"
    } else {
        ""
    };
    let clash_label = if !verdict.clash_pairs.is_empty() {
        format!(" [{} CLASH]", verdict.clash_pairs.len())
    } else {
        String::new()
    };

    lines.push(Line::from(vec![
        Span::styled(
            format!("{}: ", label),
            Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            verdict.verdict.clone(),
            Style::default()
                .fg(verdict_color(&verdict.verdict))
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(format!(
            " ({:.1}% agree / {:.1}% clash / {:.1}% neutral) gap={:+.1}pp{}",
            verdict.agree_pct, verdict.disagree_pct, verdict.neutral_pct, verdict.confidence_gap, clash_label
        )),
        Span::styled(
            format!("{}{}", confidence_label, ctx_label),
            Style::default()
                .fg(if verdict.confidence_gap > 30.0 {
                    Color::Green
                } else if verdict.confidence_gap < 10.0 {
                    Color::Yellow
                } else {
                    Color::White
                })
                .add_modifier(Modifier::BOLD),
        ),
    ]));

    if !verdict.support.is_empty() {
        let spans: Vec<Span> = std::iter::once(Span::styled(
            "  Support  : ",
            Style::default().fg(Color::White),
        ))
        .chain(verdict.support.iter().take(5).flat_map(|(token, score, reason)| {
            vec![
                Span::styled(token.clone(), Style::default().fg(score_color(*score))),
                Span::raw(format!(" ({},{:.2}) ", reason, score)),
            ]
        }))
        .collect();
        lines.push(Line::from(spans));
    }

    if !verdict.conflict.is_empty() {
        let spans: Vec<Span> = std::iter::once(Span::styled(
            "  Conflict : ",
            Style::default().fg(Color::White),
        ))
        .chain(verdict.conflict.iter().take(5).flat_map(|(token, score, reason)| {
            vec![
                Span::styled(token.clone(), Style::default().fg(score_color(*score))),
                Span::raw(format!(" ({},{:.2}) ", reason, score)),
            ]
        }))
        .collect();
        lines.push(Line::from(spans));
    }

    lines.push(Line::from(Span::styled(
        format!("  {}", verdict.explanation),
        Style::default().fg(Color::DarkGray),
    )));
    lines.push(Line::from(Span::raw("─".repeat(60))));

    lines
}

fn format_relate_result(result: &RelateResult, rsvs: &Rsvs) -> Vec<Line<'static>> {
    let mut lines = Vec::new();

    lines.push(Line::from(Span::styled(
        "Relate results",
        Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
    )));

    if result.structural_relations.is_empty() && result.related_nodes.is_empty() {
        lines.push(Line::from("  No related nodes found."));
        return lines;
    }

    if !result.structural_relations.is_empty() {
        lines.push(Line::from(Span::styled(
            "  Structural:",
            Style::default().fg(Color::White),
        )));
        for (id, score) in result.structural_relations.iter().take(10) {
            let label = rsvs
                .graph
                .get_node(*id)
                .map(|n| n.label.clone())
                .unwrap_or_else(|| format!("node_{}", id));
            lines.push(Line::from(format!(
                "    {} ({:.3})",
                label, score
            )));
        }
    }

    if !result.related_edges.is_empty() {
        lines.push(Line::from(Span::styled(
            format!("  Edges: {} found", result.related_edges.len()),
            Style::default().fg(Color::DarkGray),
        )));
    }
    lines.push(Line::from(Span::raw("─".repeat(60))));

    lines
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

fn ui(f: &mut Frame, app: &App) {
    let size = f.area();

    // Main layout: top (split view + output), middle (input), bottom (footer)
    let top_height = size.height.saturating_sub(5); // 3 for input + 1 for footer + 1 margin
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(top_height), // main area
            Constraint::Length(3),          // input bar
            Constraint::Length(1),          // footer
        ])
        .split(size);

    // Split main area into: top-half (atom browser) + bottom-half (output)
    let atom_browser_height = (top_height / 2).max(8);
    let main_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(atom_browser_height), // atom browser
            Constraint::Min(4),                     // output
        ])
        .split(chunks[0]);

    // ── Atom Browser: Left (ATOMS) | Right (SELECTED) ──
    let browser_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(28), // ATOMS panel
            Constraint::Min(20),    // SELECTED panel
        ])
        .split(main_chunks[0]);

    render_atoms_panel(f, app, browser_chunks[0]);
    render_selected_panel(f, app, browser_chunks[1]);

    // ── Output panel ──
    let output_paragraph = Paragraph::new(app.output.clone())
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(Span::styled(
                    " OUTPUT ",
                    Style::default()
                        .fg(Color::White)
                        .add_modifier(Modifier::BOLD),
                )),
        )
        .wrap(Wrap { trim: false })
        .scroll((app.output_scroll, 0));
    f.render_widget(output_paragraph, main_chunks[1]);

    // ── Input bar ──
    render_input_bar(f, app, chunks[1]);

    // ── Footer ──
    let footer = match app.mode {
        Mode::Help => Line::from(Span::styled(
            " [i]ngest [a]ppraise [c]ontext [r]elate [q]uit [?]help | Esc=cancel | Enter=submit ",
            Style::default().fg(Color::Cyan),
        )),
        _ => {
            Line::from(Span::styled(
                " [I]ngest [A]ppraise [C]ontext [R]elate [Q]uit [?]help | \u{2191}\u{2193} browse atoms | Enter=relate ",
                Style::default().fg(Color::DarkGray),
            ))
        }
    };
    f.render_widget(Paragraph::new(footer), chunks[2]);

    // ── Help overlay ──
    if app.mode == Mode::Help {
        render_help_overlay(f, size);
    }
}

fn render_atoms_panel(f: &mut Frame, app: &App, area: Rect) {
    let inner_height = area.height.saturating_sub(2) as usize; // minus borders
    let mut lines: Vec<Line<'static>> = Vec::new();

    // Calculate scroll offset to keep selected item visible
    let scroll = if app.selected_atom >= inner_height {
        app.selected_atom - inner_height + 1
    } else {
        0
    };

    let visible_range = scroll..(scroll + inner_height).min(app.atoms.len());

    for idx in visible_range {
        let atom = &app.atoms[idx];
        let (bullet, bullet_color) = tier_bullet(&atom.tier_str);
        let conf_color = confidence_color(atom.confidence);
        let bar = confidence_bar(atom.confidence, 2);

        let is_selected = idx == app.selected_atom;

        let label_display = if atom.label.len() > 10 {
            format!("{:.10}", atom.label)
        } else {
            format!("{:<10}", atom.label)
        };

        let line = if is_selected {
            Line::from(vec![
                Span::styled(
                    format!(" {} ", bullet),
                    Style::default()
                        .fg(bullet_color)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    label_display,
                    Style::default()
                        .fg(Color::White)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!("[{:.2}]", atom.confidence),
                    Style::default().fg(conf_color).add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!(" {}", bar),
                    Style::default().fg(conf_color).add_modifier(Modifier::BOLD),
                ),
            ])
        } else {
            Line::from(vec![
                Span::styled(format!(" {} ", bullet), Style::default().fg(bullet_color)),
                Span::styled(label_display, Style::default().fg(Color::White)),
                Span::styled(
                    format!("[{:.2}]", atom.confidence),
                    Style::default().fg(conf_color),
                ),
                Span::styled(format!(" {}", bar), Style::default().fg(conf_color)),
            ])
        };

        lines.push(line);
    }

    // Pad with empty lines if not enough atoms
    while lines.len() < inner_height {
        lines.push(Line::from(""));
    }

    // Navigation hint at the bottom
    if !app.atoms.is_empty() {
        let hint = format!(" \u{2191}\u{2193} nav  [Enter] relate ");
        lines.push(Line::from(Span::styled(
            hint,
            Style::default().fg(Color::DarkGray),
        )));
    }

    let atom_count = app.atoms.len();
    let title = format!(" ATOMS ({}) ", atom_count);

    let paragraph = Paragraph::new(lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(Span::styled(
                    title,
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                )),
        )
        .scroll((scroll as u16, 0));

    f.render_widget(paragraph, area);
}

fn render_selected_panel(f: &mut Frame, app: &App, area: Rect) {
    let mut lines: Vec<Line<'static>> = Vec::new();

    if let Some(atom) = app.atoms.get(app.selected_atom) {
        // Title: atom label
        lines.push(Line::from(vec![
            Span::styled(
                format!(" {} ", atom.label),
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                if atom.is_seed { "[seed]" } else { "" },
                Style::default().fg(Color::Yellow),
            ),
        ]));

        // Metadata line
        let tier_display = match atom.tier_str.as_str() {
            "stable" => "Stable",
            "candidate" => "Candidate",
            "new" => "New",
            "deprecated" => "Deprecated",
            "quarantine" => "Quarantine",
            other => other,
        };
        let (bullet, bullet_color) = tier_bullet(&atom.tier_str);
        lines.push(Line::from(vec![
            Span::raw("  Layer: "),
            Span::styled(
                format!("{}", atom.layer),
                Style::default().fg(Color::Cyan),
            ),
            Span::raw("  Tier: "),
            Span::styled(
                format!("{} {}", bullet, tier_display),
                Style::default().fg(bullet_color),
            ),
        ]));

        lines.push(Line::from(vec![
            Span::raw("  Senses: "),
            Span::styled(
                format!("{}", atom.sense_count),
                Style::default().fg(Color::White),
            ),
            Span::raw("  Conf: "),
            Span::styled(
                format!("{:.2}", atom.confidence),
                Style::default().fg(confidence_color(atom.confidence)),
            ),
            if let Some(coh) = atom.coherence {
                Span::raw(format!("  Coherence: {:.2}", coh))
            } else {
                Span::raw(String::new())
            },
        ]));

        if let Some(gs) = atom.grounding_score {
            lines.push(Line::from(vec![
                Span::raw("  Grounding: "),
                Span::styled(
                    format!("{:.2}", gs),
                    Style::default().fg(if gs > 0.5 {
                        Color::Green
                    } else {
                        Color::Yellow
                    }),
                ),
            ]));
        }

        // Compositions
        if !atom.compositions.is_empty() {
            lines.push(Line::from(Span::styled(
                "  Compositions:",
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            )));
            for (label, conf, is_seed) in atom.compositions.iter().take(8) {
                let seed_tag = if *is_seed { " (seed)" } else { "" };
                lines.push(Line::from(vec![
                    Span::styled("    \u{2192} ", Style::default().fg(Color::DarkGray)), // →
                    Span::styled(
                        format!("{:<12}", label),
                        Style::default().fg(confidence_color(*conf)),
                    ),
                    Span::styled(
                        format!("({:.2}){}", conf, seed_tag),
                        Style::default().fg(if *is_seed {
                            Color::Yellow
                        } else {
                            Color::DarkGray
                        }),
                    ),
                ]));
            }
            if atom.compositions.len() > 8 {
                lines.push(Line::from(Span::styled(
                    format!("    ... +{} more", atom.compositions.len() - 8),
                    Style::default().fg(Color::DarkGray),
                )));
            }
        } else if atom.is_seed {
            lines.push(Line::from(Span::styled(
                "  Compositions: (primitive seed)",
                Style::default().fg(Color::DarkGray),
            )));
        } else {
            lines.push(Line::from(Span::styled(
                "  Compositions: (none yet)",
                Style::default().fg(Color::DarkGray),
            )));
        }

        // Related nodes
        if !atom.related.is_empty() {
            lines.push(Line::from(Span::styled(
                "  Related:",
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            )));
            let mut related_line = String::from("   ");
            for (label, score) in atom.related.iter().take(6) {
                related_line.push_str(&format!("{}({:.2}) ", label, score));
            }
            lines.push(Line::from(Span::styled(
                related_line,
                Style::default().fg(Color::Cyan),
            )));
        }
    } else {
        lines.push(Line::from(Span::styled(
            " No atoms in graph.",
            Style::default().fg(Color::DarkGray),
        )));
        lines.push(Line::from(Span::styled(
            " Press [i] to ingest text.",
            Style::default().fg(Color::DarkGray),
        )));
    }

    let selected_label = app
        .atoms
        .get(app.selected_atom)
        .map(|a| a.label.clone())
        .unwrap_or_default();
    let title = format!(" SELECTED: {} ", selected_label);

    let paragraph = Paragraph::new(lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(Span::styled(
                    title,
                    Style::default()
                        .fg(Color::White)
                        .add_modifier(Modifier::BOLD),
                )),
        )
        .wrap(Wrap { trim: true });

    f.render_widget(paragraph, area);
}

fn render_input_bar(f: &mut Frame, app: &App, area: Rect) {
    let mode_label = format!(" {} ", app.mode);
    let input_style = match app.mode {
        Mode::Insert => Style::default().fg(Color::Green),
        Mode::Appraise => Style::default().fg(Color::Yellow),
        Mode::ContextStep1 | Mode::ContextStep2 => Style::default().fg(Color::Magenta),
        Mode::Relate => Style::default().fg(Color::Blue),
        _ => Style::default().fg(Color::White),
    };

    let input_text = vec![Line::from(vec![
        Span::styled(mode_label, input_style.add_modifier(Modifier::BOLD)),
        Span::raw("> "),
        Span::raw(app.input.clone()),
        Span::raw("_"),
    ])];
    let input_paragraph = Paragraph::new(input_text).block(Block::default().borders(Borders::ALL));
    f.render_widget(input_paragraph, area);
}

fn render_help_overlay(f: &mut Frame, size: Rect) {
    let help_lines = vec![
        Line::from(Span::styled(
            "RSVS TUI v9.0 Help",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(vec![
            Span::styled("  [i] ", Style::default().fg(Color::Green)),
            Span::raw("INSERT mode  — ingest text into the graph"),
        ]),
        Line::from(vec![
            Span::styled("  [a] ", Style::default().fg(Color::Yellow)),
            Span::raw("APPRAISE mode — evaluate statement against graph"),
        ]),
        Line::from(vec![
            Span::styled("  [c] ", Style::default().fg(Color::Magenta)),
            Span::raw("CONTEXT mode — isolated contextual appraise"),
        ]),
        Line::from(vec![
            Span::styled("      ", Style::default()),
            Span::raw("  Step 1: enter context/story"),
        ]),
        Line::from(vec![
            Span::styled("      ", Style::default()),
            Span::raw("  Step 2: enter statement to test"),
        ]),
        Line::from(vec![
            Span::styled("  [r] ", Style::default().fg(Color::Blue)),
            Span::raw("RELATE mode  — find related concepts"),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "  Atom Browser:",
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(vec![
            Span::styled("  [\u{2191}\u{2193}] ", Style::default().fg(Color::Cyan)),
            Span::raw("Navigate atoms in ATOMS panel"),
        ]),
        Line::from(vec![
            Span::styled("  [Enter] ", Style::default().fg(Color::Cyan)),
            Span::raw("Show relations for selected atom"),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("  Tier Bullets:", Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("  \u{25cf}\u{25cf}", Style::default().fg(Color::Green)),
            Span::raw(" Stable  "),
            Span::styled("\u{25cf}", Style::default().fg(Color::Yellow)),
            Span::raw(" Candidate  "),
            Span::styled("\u{25cb}", Style::default().fg(Color::Cyan)),
            Span::raw(" New  "),
            Span::styled("\u{00b7}", Style::default().fg(Color::DarkGray)),
            Span::raw(" Deprecated"),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("  [q] ", Style::default().fg(Color::Red)),
            Span::raw("Quit"),
        ]),
        Line::from(vec![
            Span::styled("  [?] ", Style::default().fg(Color::Cyan)),
            Span::raw("Toggle this help"),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "  Press ? or Esc to close help",
            Style::default().fg(Color::DarkGray),
        )),
    ];
    let help_area = centered_rect(55, 65, size);
    let help_paragraph = Paragraph::new(help_lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(Span::styled(" HELP ", Style::default().fg(Color::Cyan)))
                .style(Style::default().bg(Color::Black)),
        )
        .wrap(Wrap { trim: true });
    f.render_widget(help_paragraph, help_area);
}

fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<(), Box<dyn std::error::Error>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = App::new();
    let mut should_quit = false;

    while !should_quit {
        terminal.draw(|f| ui(f, &app))?;

        if event::poll(std::time::Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                // Only process key press events (not release on some platforms)
                if key.kind != KeyEventKind::Press {
                    continue;
                }

                match app.mode {
                    Mode::Normal => match key.code {
                        KeyCode::Char('q') => {
                            should_quit = true;
                        }
                        KeyCode::Char('?') => {
                            app.mode = Mode::Help;
                        }
                        _ => {
                            app.handle_normal_key(key.code);
                        }
                    },
                    Mode::Help => match key.code {
                        KeyCode::Char('?') | KeyCode::Esc => {
                            app.mode = Mode::Normal;
                        }
                        _ => {}
                    },
                    _ => {
                        // Input modes
                        match key.code {
                            KeyCode::Esc => {
                                app.mode = Mode::Normal;
                                app.input.clear();
                            }
                            KeyCode::Enter => {
                                let input = app.input.trim().to_string();
                                app.input.clear();
                                app.process_input(input);
                            }
                            KeyCode::Char(c) => {
                                app.input.push(c);
                            }
                            KeyCode::Backspace => {
                                app.input.pop();
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    Ok(())
}
