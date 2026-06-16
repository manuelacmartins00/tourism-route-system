# evals/analyse_benchmark.R
# ===========================================================================
# Análise estatística dos resultados do benchmark de algoritmos de optimização
# Entrada : outputs/bench_results.csv  (gerado por run_algo_benchmark.py)
# Saída   : outputs/bench_analysis/   (gráficos PDF/PNG + tabelas LaTeX)
#
# Pacotes necessários (instalar uma vez):
#   install.packages(c("tidyverse", "PMCMRplus", "scmamp", "ggplot2",
#                      "xtable", "patchwork"))
# ===========================================================================

library(tidyverse)
library(PMCMRplus)   # Friedman + Nemenyi (frdAllPairsNemenyiTest)
library(scmamp)      # Critical Difference diagram
library(ggplot2)
library(xtable)      # tabelas LaTeX
library(patchwork)   # composição de gráficos

# ── 0. Configuração ──────────────────────────────────────────────────────────
INPUT_CSV  <- "outputs/bench_results.csv"
OUT_DIR    <- "outputs/bench_analysis"
N_SEEDS    <- 20      # seeds usados no benchmark (apenas para mensagens)
ALGOS      <- c("ACO", "GA", "PSO", "GREEDY")
ALGO_COLORS <- c(ACO = "#E41A1C", GA = "#377EB8", PSO = "#4DAF4A", GREEDY = "#FF7F00")

dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
cat(sprintf("[analyse] Output: %s\n", OUT_DIR))

# ── 1. Carregar e agregar dados ───────────────────────────────────────────────
raw <- read_csv(INPUT_CSV, show_col_types = FALSE)
cat(sprintf("[analyse] %d linhas lidas (%d cenários × %d algos × %d seeds)\n",
            nrow(raw), n_distinct(raw$scenario_id), n_distinct(raw$algo), N_SEEDS))

# Média e DP por (cenário × algoritmo) — bloco para Friedman
agg <- raw %>%
  group_by(scenario_id, profile, algo) %>%
  summarise(
    mean_fitness  = mean(fitness, na.rm = TRUE),
    sd_fitness    = sd(fitness,   na.rm = TRUE),
    mean_time     = mean(elapsed_s, na.rm = TRUE),
    n_runs        = n(),
    mean_n_pois   = mean(n_pois_selected, na.rm = TRUE),
    .groups = "drop"
  )

# Tabela resumo global por algoritmo
summary_tbl <- agg %>%
  group_by(algo) %>%
  summarise(
    Média   = round(mean(mean_fitness), 2),
    DP      = round(sd(mean_fitness),   2),
    Mín     = round(min(mean_fitness),  2),
    Máx     = round(max(mean_fitness),  2),
    Tempo_s = round(mean(mean_time),    2),
    .groups = "drop"
  ) %>%
  arrange(desc(Média))

cat("\n[analyse] Tabela resumo por algoritmo:\n")
print(summary_tbl)


# ── 2. Teste de Friedman ──────────────────────────────────────────────────────
# Formato: matriz  [cenário × algoritmo]  com valores = mean_fitness
friedman_mat <- agg %>%
  select(scenario_id, algo, mean_fitness) %>%
  pivot_wider(names_from = algo, values_from = mean_fitness) %>%
  column_to_rownames("scenario_id") %>%
  as.matrix()

# Remover cenários com NA em qualquer coluna
friedman_mat <- friedman_mat[complete.cases(friedman_mat), ]
cat(sprintf("\n[analyse] Friedman: %d cenários completos\n", nrow(friedman_mat)))

ft <- friedman.test(friedman_mat)
cat(sprintf("[analyse] Friedman: chi²=%.3f, df=%d, p=%.4f\n",
            ft$statistic, ft$parameter, ft$p.value))


# ── 3. Post-hoc Nemenyi (PMCMRplus) ──────────────────────────────────────────
nemenyi <- frdAllPairsNemenyiTest(friedman_mat)
cat("\n[analyse] Nemenyi p-values (pairwise):\n")
print(round(nemenyi$p.value, 4))

# Guardar tabela LaTeX do Nemenyi
nem_df <- as.data.frame(round(nemenyi$p.value, 4))
nem_df[is.na(nem_df)] <- "—"
writeLines(
  capture.output(
    print(xtable(nem_df,
                 caption  = "Post-hoc Nemenyi: p-values par a par (correcção Bonferroni implícita no teste).",
                 label    = "tab:nemenyi"),
          type = "latex")
  ),
  file.path(OUT_DIR, "nemenyi_table.tex")
)


# ── 4. Critical Difference Diagram (scmamp) ───────────────────────────────────
# scmamp::plotCD espera: rows=problemas, cols=algoritmos, valores=ranks médios
# Calcula ranks internamente a partir da matriz de fitness
pdf(file.path(OUT_DIR, "cd_diagram.pdf"), width = 8, height = 3)
tryCatch({
  plotCD(
    results.matrix = friedman_mat,
    alpha          = 0.05,
    cex            = 1.1,
    decreasing     = TRUE   # maior fitness = melhor
  )
  title(main = "Critical Difference (Nemenyi, α=0.05)", cex.main = 0.9)
}, error = function(e) {
  cat(sprintf("[AVISO] CD diagram falhou: %s\n", e$message))
  plot.new()
  text(0.5, 0.5, "CD diagram indisponível", cex = 1.2)
})
dev.off()
cat(sprintf("[analyse] CD diagram → %s/cd_diagram.pdf\n", OUT_DIR))


# ── 5. Boxplot: distribuição de fitness por algoritmo ─────────────────────────
p_box <- ggplot(agg, aes(x = reorder(algo, -mean_fitness, FUN = median),
                          y = mean_fitness, fill = algo)) +
  geom_boxplot(alpha = 0.7, outlier.shape = 21, outlier.size = 1.5) +
  geom_jitter(width = 0.15, alpha = 0.3, size = 0.8) +
  scale_fill_manual(values = ALGO_COLORS) +
  labs(title = "Distribuição de fitness por algoritmo",
       subtitle = sprintf("%d cenários × %d seeds", nrow(friedman_mat), N_SEEDS),
       x = NULL, y = "Fitness médio por cenário",
       caption = sprintf("Friedman: p = %.4f", ft$p.value)) +
  theme_bw(base_size = 12) +
  theme(legend.position = "none")

ggsave(file.path(OUT_DIR, "boxplot_fitness.pdf"), p_box, width = 7, height = 4)
ggsave(file.path(OUT_DIR, "boxplot_fitness.png"), p_box, width = 7, height = 4, dpi = 150)
cat(sprintf("[analyse] Boxplot → %s/boxplot_fitness.pdf\n", OUT_DIR))


# ── 6. Qualidade vs. Tempo (Pareto) ──────────────────────────────────────────
pareto_df <- agg %>%
  group_by(algo) %>%
  summarise(
    fitness_med = median(mean_fitness),
    time_med    = median(mean_time),
    .groups     = "drop"
  )

p_pareto <- ggplot(pareto_df, aes(x = time_med, y = fitness_med,
                                   colour = algo, label = algo)) +
  geom_point(size = 5) +
  geom_text(vjust = -0.8, fontface = "bold", size = 4) +
  scale_colour_manual(values = ALGO_COLORS) +
  labs(title = "Qualidade vs. Tempo de execução (mediana)",
       x = "Tempo (s)", y = "Fitness mediano") +
  theme_bw(base_size = 12) +
  theme(legend.position = "none")

ggsave(file.path(OUT_DIR, "pareto_quality_time.pdf"), p_pareto, width = 6, height = 4)
ggsave(file.path(OUT_DIR, "pareto_quality_time.png"), p_pareto, width = 6, height = 4, dpi = 150)
cat(sprintf("[analyse] Pareto → %s/pareto_quality_time.pdf\n", OUT_DIR))


# ── 7. Boxplot por perfil (A–G) ───────────────────────────────────────────────
if ("profile" %in% names(agg) && n_distinct(agg$profile) > 1) {
  p_profile <- ggplot(agg, aes(x = profile, y = mean_fitness, fill = algo)) +
    geom_boxplot(position = position_dodge(0.8), alpha = 0.7, outlier.size = 0.8) +
    scale_fill_manual(values = ALGO_COLORS) +
    labs(title = "Fitness por perfil de utilizador",
         x = "Perfil", y = "Fitness médio", fill = "Algoritmo") +
    theme_bw(base_size = 11)

  ggsave(file.path(OUT_DIR, "boxplot_by_profile.pdf"), p_profile, width = 10, height = 4)
  ggsave(file.path(OUT_DIR, "boxplot_by_profile.png"), p_profile, width = 10, height = 4, dpi = 150)
  cat(sprintf("[analyse] Boxplot por perfil → %s/boxplot_by_profile.pdf\n", OUT_DIR))
}


# ── 8. Tabela LaTeX resumo ────────────────────────────────────────────────────
latex_tbl <- summary_tbl %>%
  rename(Algoritmo = algo,
         `Tempo médio (s)` = Tempo_s)

writeLines(
  capture.output(
    print(xtable(latex_tbl,
                 caption = paste0(
                   "Resumo da performance dos algoritmos de optimização — ",
                   nrow(friedman_mat), " cenários, ", N_SEEDS, " seeds independentes."),
                 label   = "tab:algo_summary",
                 digits  = c(0, 0, 2, 2, 2, 2, 2)),
          include.rownames = FALSE,
          type = "latex")
  ),
  file.path(OUT_DIR, "summary_table.tex")
)
cat(sprintf("[analyse] Tabela LaTeX → %s/summary_table.tex\n", OUT_DIR))


# ── 9. Análise de variância por algoritmo (DP médio) ─────────────────────────
var_tbl <- raw %>%
  group_by(scenario_id, algo) %>%
  summarise(sd_within = sd(fitness, na.rm = TRUE), .groups = "drop") %>%
  group_by(algo) %>%
  summarise(
    `DP médio intra-run` = round(mean(sd_within, na.rm = TRUE), 3),
    `Max DP`             = round(max(sd_within,  na.rm = TRUE), 3),
    .groups              = "drop"
  )

cat("\n[analyse] Variância interna por algoritmo (efeito da seed):\n")
print(var_tbl)

writeLines(
  capture.output(
    print(xtable(var_tbl,
                 caption = "Desvio-padrão de fitness intra-algoritmo (variância causada pela seed).",
                 label   = "tab:variance"),
          include.rownames = FALSE,
          type = "latex")
  ),
  file.path(OUT_DIR, "variance_table.tex")
)


# ── 10. Wins por algoritmo (quantas vezes foi o melhor num cenário) ──────────
wins <- agg %>%
  group_by(scenario_id) %>%
  slice_max(mean_fitness, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  count(algo, name = "wins") %>%
  mutate(pct = round(100 * wins / nrow(friedman_mat), 1)) %>%
  arrange(desc(wins))

cat("\n[analyse] Vitórias por algoritmo (melhor fitness médio num cenário):\n")
print(wins)


# ── Sumário final ─────────────────────────────────────────────────────────────
cat("\n", strrep("=", 60), "\n")
cat("SUMÁRIO\n")
cat(strrep("=", 60), "\n")
cat(sprintf("Cenários analisados : %d\n", nrow(friedman_mat)))
cat(sprintf("Seeds por cenário   : %d\n", N_SEEDS))
cat(sprintf("Teste de Friedman   : chi²=%.3f, p=%.4f\n", ft$statistic, ft$p.value))
cat(sprintf("Significância (<0.05): %s\n",
            ifelse(ft$p.value < 0.05, "SIM — diferenças significativas", "NÃO")))
cat("\nOutputs gerados:\n")
for (f in list.files(OUT_DIR, full.names = TRUE)) cat(sprintf("  %s\n", f))
