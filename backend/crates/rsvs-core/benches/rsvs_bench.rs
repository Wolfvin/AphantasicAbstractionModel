use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rsvs::{jaccard_sets, CoocStats, PipelineConfig, Rsvs, SenseConfig, SenseManager};

fn bench_jaccard(c: &mut Criterion) {
    let set_a: Vec<u32> = (1..100).collect();
    let set_b: Vec<u32> = (50..150).collect();
    c.bench_function("jaccard_100_elements", |bencher| {
        bencher.iter(|| jaccard_sets(&set_a, &set_b))
    });
}

fn bench_npmi(c: &mut Criterion) {
    let mut stats = CoocStats::new();
    for _ in 0..10 {
        stats.ingest_sentence(&[
            "stone".into(),
            "hard".into(),
            "solid".into(),
            "rough".into(),
        ]);
    }
    for _ in 0..5 {
        stats.ingest_sentence(&["water".into(), "liquid".into(), "clear".into()]);
    }
    c.bench_function("npmi_lookup", |bencher| {
        bencher.iter(|| stats.npmi(black_box("stone"), black_box("hard")))
    });
}

fn bench_cooc_ingest(c: &mut Criterion) {
    let tokens: Vec<String> = (0..20).map(|i| format!("token_{}", i)).collect();
    c.bench_function("cooc_ingest_sentence_20_tokens", |bencher| {
        bencher.iter(|| {
            let mut stats = CoocStats::new();
            stats.ingest_sentence(&tokens)
        })
    });
}

fn bench_sense_ingest(c: &mut Criterion) {
    let mut sm = SenseManager::new(SenseConfig::default());
    // Warm up one sense first
    sm.ingest(vec![1, 2, 3, 4, 5]);
    c.bench_function("sense_ingest_10_atoms", |bencher| {
        bencher.iter(|| sm.ingest(black_box(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10])))
    });
}

fn bench_pipeline_ingest(c: &mut Criterion) {
    let mut rsvs = Rsvs::new(PipelineConfig::default());
    let text = "Stone is hard and solid. Rock is heavy and rough. \
                Water is clear and liquid. Metal conducts heat well.";
    c.bench_function("pipeline_ingest_text", |bencher| {
        bencher.iter(|| rsvs.ingest_text(black_box(text)))
    });
}

criterion_group!(
    benches,
    bench_jaccard,
    bench_npmi,
    bench_cooc_ingest,
    bench_sense_ingest,
    bench_pipeline_ingest
);
criterion_main!(benches);
