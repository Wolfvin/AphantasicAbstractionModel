//! RSVS TUI — Terminal User Interface for RSVS v8.3
//!
//! Interactive terminal interface with modes:
//! - NORMAL: navigate with keyboard shortcuts
//! - INSERT: ingest text into the graph
//! - APPRAISE: evaluate statement against entire graph
//! - CONTEXT: contextual appraise (isolated, graph untouched)
//! - RELATE: find related nodes for a concept

use crossterm::{
    event::{self, Event, KeyCode},
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
use rsvs::{AppraiseResult, AppraiseVerdict, IngestStats, PipelineConfig, RelateResult, Rsvs};
use rsvs::session::SessionGraph;
use std::io;

// ---------------------------------------------------------------------------
// App state
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum Mode {
    Normal,
    Insert,
    Appraise,
    ContextStep1, // waiting for context input
    ContextStep2, // waiting for statement input
    Relate,
    Help,
}

impl std::fmt::Display for Mode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Mode::Normal => write!(f, "NORMAL"),
            Mode::Insert => write!(f, "INSERT"),
            Mode::Appraise => write!(f, "APPRAISE"),
            Mode::ContextStep1 => write!(f, "CONTEXT(step1)"),
            Mode::ContextStep2 => write!(f, "CONTEXT(step2)"),
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
    context_buffer: String, // holds context from step 1 of CONTEXT mode
    scroll_offset: u16,
}

impl App {
    fn new() -> Self {
        let config = PipelineConfig {
            entity_promote_n: 3,
            ..PipelineConfig::default()
        };
        let rsvs = Rsvs::new(config).expect("Failed to initialize RSVS");
        Self {
            rsvs,
            mode: Mode::Normal,
            input: String::new(),
            output: vec![Line::from(Span::styled(
                "RSVS v8.3 TUI ready. Press ? for help.",
                Style::default().fg(Color::Cyan),
            ))],
            context_buffer: String::new(),
            scroll_offset: 0,
        }
    }

    fn graph_status_lines(&self) -> Vec<Line<'static>> {
        let status = self.rsvs.status();
        let atom_count = status.total_atoms;
        let edge_count = self.rsvs.graph.edge_count();
        let node_count = status.total_nodes;
        let ctx_count = status.total_contexts;
        vec![
            Line::from(Span::styled(
                "GRAPH STATUS",
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
            )),
            Line::from(format!("  atoms:     {}", atom_count)),
            Line::from(format!("  nodes:     {}", node_count)),
            Line::from(format!("  edges:     {}", edge_count)),
            Line::from(format!("  contexts:  {}", ctx_count)),
            Line::from(format!("  mode:      {}", self.mode)),
        ]
    }

    fn push_output(&mut self, lines: Vec<Line<'static>>) {
        self.output.extend(lines);
        // Auto-scroll to bottom
        self.scroll_offset = 0;
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
                } else {
                    self.context_buffer = input;
                    self.push_output(vec![Line::from(Span::styled(
                        "Context saved. Now enter the statement to test:",
                        Style::default().fg(Color::Cyan),
                    ))]);
                    self.mode = Mode::ContextStep2;
                    return; // Don't go back to Normal yet
                }
                self.mode = Mode::Normal;
            }
            Mode::ContextStep2 => {
                if input.is_empty() {
                    self.push_output(vec![Line::from(Span::styled(
                        "Empty statement — nothing appraised.",
                        Style::default().fg(Color::Yellow),
                    ))]);
                } else {
                    // Use SessionGraph for Dual Memory pattern
                    match SessionGraph::new(&self.context_buffer, self.rsvs.config.clone()) {
                        Ok(session) => {
                            // Show session stats
                            let sstats = session.stats();
                            self.push_output(vec![Line::from(Span::styled(
                                format!("[Session] {} sentences ingested → {} atoms induced",
                                    sstats.sentences_ingested, sstats.atoms_induced),
                                Style::default().fg(Color::DarkGray),
                            ))]);
                            // Show truncated context
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
                            // Appraise with verbose verdict
                            let verdict = session.appraise(&input);
                            self.push_output(format_verdict_result(&verdict, "Contextual Appraise"));
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
                    "APPRAISE mode — type statement to evaluate, Enter to submit, Esc to cancel:",
                    Style::default().fg(Color::Green),
                ))]);
            }
            KeyCode::Char('c') => {
                self.mode = Mode::ContextStep1;
                self.context_buffer.clear();
                self.push_output(vec![Line::from(Span::styled(
                    "CONTEXT mode — Step 1: enter story/context, Enter to submit, Esc to cancel:",
                    Style::default().fg(Color::Green),
                ))]);
            }
            KeyCode::Char('r') => {
                self.mode = Mode::Relate;
                self.push_output(vec![Line::from(Span::styled(
                    "RELATE mode — type concept to find relations, Enter to submit, Esc to cancel:",
                    Style::default().fg(Color::Green),
                ))]);
            }
            KeyCode::Char('?') => {
                if self.mode == Mode::Help {
                    self.mode = Mode::Normal;
                } else {
                    self.mode = Mode::Help;
                }
            }
            KeyCode::Char('q') => {
                // Will be handled in main loop
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

fn format_ingest_result(stats: &IngestStats) -> Vec<Line<'static>> {
    vec![
        Line::from(Span::styled(
            "Ingest complete",
            Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
        )),
        Line::from(format!("  sentences:    {}", stats.sentences_processed)),
        Line::from(format!("  atoms promoted: {}", stats.atoms_promoted)),
        Line::from(format!("  senses created: {}", stats.sense_created)),
        Line::from(format!("  senses assigned: {}", stats.sense_assigned)),
        Line::from(format!("  confidence updated: {}", stats.confidence_updated)),
        Line::from(format!("  compositions induced: {}", stats.compositions_induced)),
    ]
}

fn verdict_color(verdict: &str) -> Color {
    match verdict {
        "consistent" => Color::Green,
        "partial" => Color::Yellow,
        "novel" => Color::Cyan,
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

fn format_verdict_result(verdict: &AppraiseVerdict, label: &str) -> Vec<Line<'static>> {
    let mut lines = Vec::new();

    // Confidence label — v7.5: gap is now agree - genuine_clash
    let confidence_label = if verdict.confidence_gap > 30.0 {
        " [CONFIDENT]"
    } else if verdict.confidence_gap > 10.0 {
        " [MODERATE]"
    } else if verdict.confidence_gap > 0.0 {
        " [AMBIGUOUS]"
    } else {
        " [INVERTED]"
    };
    let ctx_label = if verdict.is_contextual { " [CONTEXTUAL]" } else { "" };
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
            Style::default().fg(
                if verdict.confidence_gap > 30.0 { Color::Green }
                else if verdict.confidence_gap < 10.0 { Color::Yellow }
                else { Color::White }
            ).add_modifier(Modifier::BOLD),
        ),
    ]));

    // Support evidence with reasons
    if !verdict.support.is_empty() {
        let spans: Vec<Span> = std::iter::once(Span::styled(
            "  Support  : ",
            Style::default().fg(Color::White),
        ))
        .chain(verdict.support.iter().take(5).flat_map(|(token, score, reason)| {
            vec![
                Span::styled(
                    token.clone(),
                    Style::default().fg(score_color(*score)),
                ),
                Span::raw(format!(" ({},{:.2}) ", reason, score)),
            ]
        }))
        .collect();
        lines.push(Line::from(spans));
    }

    // Conflict evidence with reasons
    if !verdict.conflict.is_empty() {
        let spans: Vec<Span> = std::iter::once(Span::styled(
            "  Conflict : ",
            Style::default().fg(Color::White),
        ))
        .chain(verdict.conflict.iter().take(5).flat_map(|(token, score, reason)| {
            vec![
                Span::styled(
                    token.clone(),
                    Style::default().fg(score_color(*score)),
                ),
                Span::raw(format!(" ({},{:.2}) ", reason, score)),
            ]
        }))
        .collect();
        lines.push(Line::from(spans));
    }

    // Explanation
    lines.push(Line::from(Span::styled(
        format!("  {}", verdict.explanation),
        Style::default().fg(Color::DarkGray),
    )));

    // Separator line
    lines.push(Line::from(Span::raw("─".repeat(60))));

    lines
}

fn format_appraise_result(result: &AppraiseResult, label: &str) -> Vec<Line<'static>> {
    let mut lines = Vec::new();

    lines.push(Line::from(vec![
        Span::styled(
            format!("{}: ", label),
            Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            result.verdict.clone(),
            Style::default()
                .fg(verdict_color(&result.verdict))
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(format!(
            " ({:.1}% agree / {:.1}% clash / {:.1}% neutral)",
            result.agree_pct, result.disagree_pct, result.neutral_pct
        )),
    ]));

    // Evidence: support (positive score) and conflict (negative score)
    let support: Vec<&(String, f32)> = result.evidence.iter().filter(|(_, s)| *s > 0.0).collect();
    let conflict: Vec<&(String, f32)> = result.evidence.iter().filter(|(_, s)| *s <= 0.0).collect();

    if !support.is_empty() {
        let spans: Vec<Span> = std::iter::once(Span::styled(
            "  Support  : ",
            Style::default().fg(Color::White),
        ))
        .chain(support.iter().flat_map(|(token, score)| {
            vec![
                Span::styled(
                    token.clone(),
                    Style::default().fg(score_color(*score)),
                ),
                Span::raw(format!(" ({:.2}) ", score)),
            ]
        }))
        .collect();
        lines.push(Line::from(spans));
    }

    if !conflict.is_empty() {
        let spans: Vec<Span> = std::iter::once(Span::styled(
            "  Conflict : ",
            Style::default().fg(Color::White),
        ))
        .chain(conflict.iter().flat_map(|(token, score)| {
            vec![
                Span::styled(
                    token.clone(),
                    Style::default().fg(score_color(*score)),
                ),
                Span::raw(format!(" ({:.2}) ", score)),
            ]
        }))
        .collect();
        lines.push(Line::from(spans));
    }

    // Convergence info
    if !result.convergence_info.is_empty() {
        let spans: Vec<Span> = std::iter::once(Span::styled(
            "  Converge : ",
            Style::default().fg(Color::DarkGray),
        ))
        .chain(result.convergence_info.iter().flat_map(|(label, boost)| {
            vec![
                Span::raw(format!("{} ({:.2}) ", label, boost)),
            ]
        }))
        .collect();
        lines.push(Line::from(spans));
    }

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

    // Structural relations (most relevant)
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

    // Related edges
    if !result.related_edges.is_empty() {
        lines.push(Line::from(Span::styled(
            format!("  Edges: {} found", result.related_edges.len()),
            Style::default().fg(Color::DarkGray),
        )));
    }

    lines
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

fn ui(f: &mut Frame, app: &App) {
    let size = f.area();

    // Main layout: top (status + output), middle (input), bottom (footer)
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(5),    // main area (status + output)
            Constraint::Length(3), // input bar
            Constraint::Length(1), // footer
        ])
        .split(size);

    // Split main area into left (status) and right (output)
    let main_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(24), // status panel
            Constraint::Min(10),    // output panel
        ])
        .split(chunks[0]);

    // --- Left panel: Graph Status ---
    let status_lines = app.graph_status_lines();
    let status_paragraph = Paragraph::new(status_lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(Span::styled(
                    " RSVS v8.3 ",
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                )),
        )
        .wrap(Wrap { trim: true });
    f.render_widget(status_paragraph, main_chunks[0]);

    // --- Right panel: Output ---
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
        .scroll((app.scroll_offset, 0));
    f.render_widget(output_paragraph, main_chunks[1]);

    // --- Input bar ---
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
    let input_paragraph = Paragraph::new(input_text)
        .block(Block::default().borders(Borders::ALL));
    f.render_widget(input_paragraph, chunks[1]);

    // --- Footer ---
    let footer = match app.mode {
        Mode::Help => Line::from(Span::styled(
            " [i]ngest [a]ppraise [c]ontext [r]elate [q]uit [?]help | Esc=cancel | Enter=submit ",
            Style::default().fg(Color::Cyan),
        )),
        _ => {
            let shortcuts = " [I]ngest [A]ppraise [C]ontext [R]elate [Q]uit [?]help ";
            Line::from(Span::styled(shortcuts, Style::default().fg(Color::DarkGray)))
        }
    };
    let footer_paragraph = Paragraph::new(footer);
    f.render_widget(footer_paragraph, chunks[2]);

    // --- Help overlay ---
    if app.mode == Mode::Help {
        let help_lines = vec![
            Line::from(Span::styled(
                "RSVS TUI Help",
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
        let help_area = centered_rect(50, 60, size);
        let help_paragraph = Paragraph::new(help_lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(Span::styled(
                        " HELP ",
                        Style::default().fg(Color::Cyan),
                    ))
                    .style(Style::default().bg(Color::Black)),
            )
            .wrap(Wrap { trim: true });
        f.render_widget(help_paragraph, help_area);
    }
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
    // Setup terminal
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
                match app.mode {
                    Mode::Normal => {
                        match key.code {
                            KeyCode::Char('q') => {
                                should_quit = true;
                            }
                            KeyCode::Char('?') => {
                                if app.mode == Mode::Help {
                                    app.mode = Mode::Normal;
                                } else {
                                    app.mode = Mode::Help;
                                }
                            }
                            _ => {
                                app.handle_normal_key(key.code);
                            }
                        }
                    }
                    Mode::Help => {
                        match key.code {
                            KeyCode::Char('?') | KeyCode::Esc => {
                                app.mode = Mode::Normal;
                            }
                            _ => {}
                        }
                    }
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

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen
    )?;
    terminal.show_cursor()?;

    Ok(())
}
