# Next Step RSVS

## 1) Keputusan Teknis
- Loop RSVS secara mekanisme berfungsi (sense mismatch bisa terdeteksi).
- Bottleneck utama saat ini adalah data quality + data size, bukan arsitektur.
- Verifier keyword-level harus diganti ke sense-level verification.
- Prefix runtime untuk dLLM harus sense-aware, bukan entity-only.
- Untuk prototipe, pakai existing LLM sebagai placeholder dulu; custom dLLM belakangan.

## 2) Fitur Baru: Quality Manager (Rule Input + Policy Verification)
- Tambah modul **Quality Manager** sebagai lapisan verifikasi untuk domain aturan ketat (contoh: pajak).
- User bisa **input aturan/peraturan** ke sistem (rule, exception, effective date, source).
- LLM tetap generate jawaban, tetapi keputusan akhir melewati RSVS checker:
  - `approved`
  - `approved_with_risk`
  - `rejected`
  - `needs_human_review`
- Output wajib punya traceability:
  - aturan mana yang cocok,
  - aturan mana yang dilanggar,
  - alasan verdict.
- Arsitektur target v1:
  - `User Prompt -> LLM Generation -> RSVS Quality Manager -> Verdict + Reason`
- Scope awal difokuskan ke subset aturan sempit dulu (MVP), bukan seluruh regulasi sekaligus.

## 3) Action Items (1 Minggu)
1. Naikkan corpus dari 150 -> minimal 1k+ sentence per concept/domain kunci.
2. Pisahkan dataset train vs eval agar hasil loop bisa dibandingkan objektif.
3. Implement verifier v2:
- target sense dari prefix,
- active sense dari output,
- reject jika sense mismatch.
4. Tambah metrik evaluasi:
- sense consistency rate,
- contradiction rejection rate,
- false reject rate.
5. Jalankan benchmark lintas domain (geology, biology, water, materials) dan simpan baseline JSON.
6. Tuning konservatif threshold (`entity_promote_n`, `min_cooc`, `theta_assign`) berdasarkan hasil benchmark.
7. Freeze format prefix v1:
- `<RSVS sense=... core=... context=...> entity </RSVS>`.
8. Mulai Quality Manager MVP:
- definisikan format input aturan user,
- simpan aturan terstruktur + version/effective date,
- tambah verdict classifier + reason generator berbasis RSVS event trace.

## 4) Checklist Eksperimen Lanjutan
- [ ] Uji "water liquid" vs output yang mengarah ke ice/material.
- [ ] Uji "solid material" vs konteks phase-change (solid/liquid) untuk cek sense contamination.
- [ ] Uji "rock geology" vs "rock music" (kalau data cukup) untuk multi-sense separation.
- [ ] Bandingkan hasil sebelum/sesudah corpus expansion.
- [ ] Validasi event trace: intended_sense, activated_sense, decision, reason.
- [ ] Pastikan metrik stabil di 3 run berulang (repeatability).
- [ ] Uji Quality Manager dengan 10 aturan contoh + 20 kasus uji (pass/fail/mixed).
- [ ] Verifikasi bahwa setiap verdict mengandung referensi aturan yang jelas.

## Merge Note: RSVS UI Subtree
- Date: 2026-04-22
- Source archive: `workspace-faaa4590-3440-4b0b-8f4d-e17b043d370a.tar`
- Imported into: `apps/rsvs-ui`
- Scope: Next.js RSVS interface (3D graph UI, zustand stores, components, mock data)
- Non-goal: no overwrite of Rust/Python core layout in repository root

## Migration Progress (UI)
- UI subtree bootstrap started at `apps/rsvs-ui`.
- Non-essential imported artifacts have been trimmed.
- Makefile now includes `ui-install`, `ui-dev`, `ui-lint`, `ui-build`.
