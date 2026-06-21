# evals/analyse_benchmark.R
# ===========================================================================
# Análise estatística dos resultados do benchmark de algoritmos (grid search)
# Entrada : outputs/bench_results.csv  (gerado por run_algo_benchmark.py)
# Saída   : outputs/bench_analysis/   (gráficos PDF/PNG + tabelas LaTeX)
#
# Estrutura do CSV:
#   algo ∈ {ACO_S, ACO_L, GA_S, GA_L, PSO_S, PSO_L, GREEDY}
#   componentes: time_efficiency, proximity_component, diversity_component,
#                distance_penalty, cat_indata_comp, cat_general_comp
#
# Pacotes necessários (instalar uma vez):
#   install.packages(c("tidyverse", "PMCMRplus", "scmamp", "ggplot2",
#                      "xtable", "patchwork"))
# ===========================================================================

library(tidyverse)
library(PMCMRplus)   # Friedman + Nemenyi
library(scmamp)      # CD diagram
library(ggplot2)
library(xtable)
library(patchwork)

# ── 0. Configuração ──────────────────────────────────────────────────────────
INPUT_CSV <- "outputs/bench_results.csv"
OUT_DIR   <- "outputs/bench_analysis"
N_SEEDS   <- 5

ALGO_FAMILIES <- c("ACO", "GA", "PSO", "GREEDY")
COMPONENTS    <- c("time_efficiency", "proximity_component", "diversity_component",
                   "distance_penalty", "cat_indata_comp", "cat_general_comp")
COMP_LABELS   <- c("Time Utilization", "Geographic Proximity", "Diversity",
                   "Distance Penalty", "Category (In-Data)", "Category (General)")

FAMILY_COLORS <- c(ACO="#E41A1C", GA="#377EB8", PSO="#4DAF4A", GREEDY="#FF7F00")
CONFIG_COLORS <- c(ACO_S="#FF9999", ACO_L="#E41A1C",
                   GA_S="#99C2F0",  GA_L="#377EB8",
                   PSO_S="#99E099", PSO_L="#4DAF4A",
                   GREEDY="#FF7F00")

dir.create(OUT_DIR, showWarnings=FALSE, recursive=TRUE)
cat(sprintf("[analyse] Output: %s\n", OUT_DIR))


# ── 1. Carregar e preparar dados ─────────────────────────────────────────────
raw <- read_csv(INPUT_CSV, show_col_types=FALSE)
cat(sprintf("[analyse] %d linhas | %d cenários | %d configs | %d seeds\n",
            nrow(raw), n_distinct(raw$scenario_id),
            n_distinct(raw$algo), N_SEEDS))

# Adicionar coluna de família (ACO_S → ACO, GREEDY → GREEDY)
raw <- raw %>%
  mutate(family = str_remove(algo, "_[SL]$"))

# Agregar por (cenário × config) — média sobre seeds
agg_config <- raw %>%
  group_by(scenario_id, profile, algo, family) %>%
  summarise(
    mean_fitness       = mean(fitness,             na.rm=TRUE),
    sd_fitness         = sd(fitness,               na.rm=TRUE),
    mean_elapsed       = mean(elapsed_s,           na.rm=TRUE),
    mean_n_pois        = mean(n_pois_selected,     na.rm=TRUE),
    mean_time_eff      = mean(time_efficiency,     na.rm=TRUE),
    mean_proximity     = mean(proximity_component, na.rm=TRUE),
    mean_diversity     = mean(diversity_component, na.rm=TRUE),
    mean_dist_pen      = mean(distance_penalty,    na.rm=TRUE),
    mean_cat_indata    = mean(cat_indata_comp,     na.rm=TRUE),
    mean_cat_general   = mean(cat_general_comp,    na.rm=TRUE),
    .groups="drop"
  )


# ── 2. Grid search: melhor config por (cenário × família) ────────────────────
best_per_family <- agg_config %>%
  group_by(scenario_id, profile, family) %>%
  slice_max(mean_fitness, n=1, with_ties=FALSE) %>%
  ungroup() %>%
  rename(algo=algo, fitness_best=mean_fitness)

# Qual config ganhou mais vezes por família?
config_wins <- agg_config %>%
  group_by(scenario_id, family) %>%
  slice_max(mean_fitness, n=1, with_ties=FALSE) %>%
  ungroup() %>%
  count(family, algo, name="wins") %>%
  group_by(family) %>%
  mutate(pct=round(100*wins/sum(wins),1)) %>%
  arrange(family, desc(wins))

cat("\n[analyse] Config wins por família (qual S ou L ganhou mais):\n")
print(config_wins)

writeLines(
  capture.output(
    print(xtable(config_wins,
                 caption="Grid search: configuração vencedora por família de algoritmo.",
                 label="tab:config_wins"),
          include.rownames=FALSE, type="latex")
  ),
  file.path(OUT_DIR, "config_wins_table.tex")
)


# ── 3. Friedman (melhor config por família) ───────────────────────────────────
friedman_mat <- best_per_family %>%
  select(scenario_id, family, fitness_best) %>%
  pivot_wider(names_from=family, values_from=fitness_best) %>%
  column_to_rownames("scenario_id") %>%
  select(all_of(ALGO_FAMILIES)) %>%
  as.matrix()

friedman_mat <- friedman_mat[complete.cases(friedman_mat), ]
cat(sprintf("\n[analyse] Friedman: %d cenários completos\n", nrow(friedman_mat)))

ft <- friedman.test(friedman_mat)
cat(sprintf("[analyse] chi²=%.3f  df=%d  p=%.6f\n",
            ft$statistic, ft$parameter, ft$p.value))


# ── 4. Nemenyi post-hoc ───────────────────────────────────────────────────────
nemenyi <- frdAllPairsNemenyiTest(friedman_mat)
cat("\n[analyse] Nemenyi p-values:\n")
print(round(nemenyi$p.value, 4))

nem_df <- as.data.frame(round(nemenyi$p.value, 4))
nem_df[is.na(nem_df)] <- "—"
writeLines(
  capture.output(
    print(xtable(nem_df,
                 caption="Post-hoc Nemenyi: p-values par a par (\\alpha=0.05).",
                 label="tab:nemenyi"),
          type="latex")
  ),
  file.path(OUT_DIR, "nemenyi_table.tex")
)


# ── 5. CD Diagram ─────────────────────────────────────────────────────────────
pdf(file.path(OUT_DIR, "cd_diagram.pdf"), width=8, height=3)
tryCatch({
  plotCD(results.matrix=friedman_mat, alpha=0.05, cex=1.1, decreasing=TRUE)
  title(main=sprintf("Critical Difference — Nemenyi α=0.05  |  N=%d cenários",
                     nrow(friedman_mat)), cex.main=0.9)
}, error=function(e) {
  cat(sprintf("[AVISO] CD diagram: %s\n", e$message))
  plot.new(); text(0.5, 0.5, "CD diagram indisponível")
})
dev.off()
cat(sprintf("[analyse] CD diagram → %s/cd_diagram.pdf\n", OUT_DIR))


# ── 6. Boxplot: fitness por família (melhor config) ───────────────────────────
agg_fam <- best_per_family %>%
  rename(mean_fitness=fitness_best)

p_box_fam <- ggplot(agg_fam,
                    aes(x=reorder(family, -mean_fitness, FUN=median),
                        y=mean_fitness, fill=family)) +
  geom_boxplot(alpha=0.7, outlier.shape=21, outlier.size=1.5) +
  geom_jitter(width=0.15, alpha=0.25, size=0.7) +
  scale_fill_manual(values=FAMILY_COLORS) +
  labs(title="Fitness por família de algoritmo (melhor configuração)",
       subtitle=sprintf("%d cenários × %d seeds | Friedman p=%.4f",
                        nrow(friedman_mat), N_SEEDS, ft$p.value),
       x=NULL, y="Fitness médio por cenário") +
  theme_bw(base_size=12) + theme(legend.position="none")

ggsave(file.path(OUT_DIR, "boxplot_family_fitness.pdf"), p_box_fam, width=7, height=4)
ggsave(file.path(OUT_DIR, "boxplot_family_fitness.png"), p_box_fam, width=7, height=4, dpi=150)


# ── 7. Boxplot: S vs L por família ────────────────────────────────────────────
p_box_cfg <- ggplot(agg_config %>% filter(family != "GREEDY"),
                    aes(x=algo, y=mean_fitness, fill=algo)) +
  geom_boxplot(alpha=0.7, outlier.shape=21, outlier.size=1.2) +
  scale_fill_manual(values=CONFIG_COLORS) +
  facet_wrap(~family, scales="free_x", nrow=1) +
  labs(title="Grid search: configuração leve (S) vs. produção (L)",
       x=NULL, y="Fitness médio por cenário") +
  theme_bw(base_size=11) + theme(legend.position="none")

ggsave(file.path(OUT_DIR, "boxplot_config_SvsL.pdf"), p_box_cfg, width=10, height=4)
ggsave(file.path(OUT_DIR, "boxplot_config_SvsL.png"), p_box_cfg, width=10, height=4, dpi=150)
cat("[analyse] Boxplots fitness → bench_analysis/\n")


# ── 8. Análise por componente ─────────────────────────────────────────────────
comp_long <- agg_config %>%
  select(scenario_id, family, starts_with("mean_")) %>%
  pivot_longer(cols=c(mean_time_eff, mean_proximity, mean_diversity,
                      mean_dist_pen, mean_cat_indata, mean_cat_general),
               names_to="component", values_to="value") %>%
  mutate(component=recode(component,
    mean_time_eff   = "Time Utilization",
    mean_proximity  = "Geographic Proximity",
    mean_diversity  = "Diversity",
    mean_dist_pen   = "Distance Penalty",
    mean_cat_indata = "Category (In-Data)",
    mean_cat_general= "Category (General)"
  ))

# Best config per family per scenario for components
best_comp <- agg_config %>%
  group_by(scenario_id, family) %>%
  slice_max(mean_fitness, n=1, with_ties=FALSE) %>%
  ungroup() %>%
  select(scenario_id, family,
         mean_time_eff, mean_proximity, mean_diversity,
         mean_dist_pen, mean_cat_indata, mean_cat_general) %>%
  pivot_longer(cols=starts_with("mean_"),
               names_to="component", values_to="value") %>%
  mutate(component=recode(component,
    mean_time_eff   = "Time Utilization",
    mean_proximity  = "Geographic Proximity",
    mean_diversity  = "Diversity",
    mean_dist_pen   = "Distance Penalty",
    mean_cat_indata = "Category (In-Data)",
    mean_cat_general= "Category (General)"
  ))

p_comp <- ggplot(best_comp,
                 aes(x=reorder(family, -value, FUN=median),
                     y=value, fill=family)) +
  geom_boxplot(alpha=0.7, outlier.shape=21, outlier.size=0.8) +
  scale_fill_manual(values=FAMILY_COLORS) +
  facet_wrap(~component, scales="free_y", ncol=3) +
  labs(title="Componentes de fitness por algoritmo (melhor config)",
       x=NULL, y="Valor [0–100]") +
  theme_bw(base_size=10) +
  theme(legend.position="none",
        axis.text.x=element_text(angle=30, hjust=1))

ggsave(file.path(OUT_DIR, "components_by_algo.pdf"), p_comp, width=12, height=7)
ggsave(file.path(OUT_DIR, "components_by_algo.png"), p_comp, width=12, height=7, dpi=150)
cat("[analyse] Componentes → bench_analysis/components_by_algo.*\n")

# Tabela resumo das componentes por família
comp_summary <- best_comp %>%
  group_by(family, component) %>%
  summarise(mean=round(mean(value, na.rm=TRUE), 2),
            sd=round(sd(value, na.rm=TRUE), 2), .groups="drop") %>%
  mutate(label=sprintf("%.2f ± %.2f", mean, sd)) %>%
  select(family, component, label) %>%
  pivot_wider(names_from=component, values_from=label)

writeLines(
  capture.output(
    print(xtable(comp_summary,
                 caption="Média ± DP de cada componente de fitness por algoritmo.",
                 label="tab:components"),
          include.rownames=FALSE, type="latex")
  ),
  file.path(OUT_DIR, "components_table.tex")
)


# ── 9. Qualidade vs. Tempo (Pareto) ───────────────────────────────────────────
pareto_df <- best_per_family %>%
  group_by(family) %>%
  summarise(fitness_med=median(fitness_best, na.rm=TRUE),
            time_med=median(mean_elapsed, na.rm=TRUE), .groups="drop")

p_pareto <- ggplot(pareto_df, aes(x=time_med, y=fitness_med,
                                   colour=family, label=family)) +
  geom_point(size=6) +
  geom_text(vjust=-1, fontface="bold", size=4) +
  scale_colour_manual(values=FAMILY_COLORS) +
  labs(title="Qualidade vs. Tempo de execução (mediana)",
       x="Tempo médio por run (s)", y="Fitness mediano") +
  theme_bw(base_size=12) + theme(legend.position="none")

ggsave(file.path(OUT_DIR, "pareto_quality_time.pdf"), p_pareto, width=6, height=4)
ggsave(file.path(OUT_DIR, "pareto_quality_time.png"), p_pareto, width=6, height=4, dpi=150)


# ── 10. Boxplot por perfil ────────────────────────────────────────────────────
if (n_distinct(agg_fam$profile) > 1) {
  p_profile <- ggplot(agg_fam, aes(x=profile, y=mean_fitness, fill=family)) +
    geom_boxplot(position=position_dodge(0.8), alpha=0.7, outlier.size=0.8) +
    scale_fill_manual(values=FAMILY_COLORS) +
    labs(title="Fitness por perfil de utilizador e algoritmo",
         x="Perfil", y="Fitness médio", fill="Algoritmo") +
    theme_bw(base_size=11)

  ggsave(file.path(OUT_DIR, "boxplot_by_profile.pdf"), p_profile, width=11, height=4)
  ggsave(file.path(OUT_DIR, "boxplot_by_profile.png"), p_profile, width=11, height=4, dpi=150)
}


# ── 11. Variância interna (efeito seed) ───────────────────────────────────────
var_tbl <- raw %>%
  group_by(scenario_id, algo) %>%
  summarise(sd_within=sd(fitness, na.rm=TRUE), .groups="drop") %>%
  group_by(algo) %>%
  summarise(`DP médio (seed)`=round(mean(sd_within, na.rm=TRUE), 3),
            `Max DP`=round(max(sd_within, na.rm=TRUE), 3), .groups="drop") %>%
  arrange(desc(`DP médio (seed)`))

cat("\n[analyse] Variância interna por config (efeito seed):\n")
print(var_tbl)

writeLines(
  capture.output(
    print(xtable(var_tbl,
                 caption="Desvio-padrão de fitness intra-configuração (variância pela seed).",
                 label="tab:variance"),
          include.rownames=FALSE, type="latex")
  ),
  file.path(OUT_DIR, "variance_table.tex")
)


# ── 12. Vitórias por família ──────────────────────────────────────────────────
wins <- agg_fam %>%
  group_by(scenario_id) %>%
  slice_max(mean_fitness, n=1, with_ties=FALSE) %>%
  ungroup() %>%
  count(family, name="wins") %>%
  mutate(pct=round(100*wins/nrow(friedman_mat), 1)) %>%
  arrange(desc(wins))

cat("\n[analyse] Vitórias por família:\n")
print(wins)


# ── 13. Tabela LaTeX resumo global ───────────────────────────────────────────
summary_tbl <- agg_fam %>%
  group_by(family) %>%
  summarise(
    Média   = round(mean(mean_fitness),  2),
    DP      = round(sd(mean_fitness),    2),
    Mín     = round(min(mean_fitness),   2),
    Máx     = round(max(mean_fitness),   2),
    .groups = "drop"
  ) %>%
  left_join(wins %>% select(family, wins, pct), by="family") %>%
  arrange(desc(Média)) %>%
  rename(Algoritmo=family, Vitórias=wins, `%`=pct)

writeLines(
  capture.output(
    print(xtable(summary_tbl,
                 caption=sprintf(
                   "Resumo de performance — %d cenários, %d seeds, melhor config por família.",
                   nrow(friedman_mat), N_SEEDS),
                 label="tab:algo_summary", digits=c(0,0,2,2,2,2,0,1)),
          include.rownames=FALSE, type="latex")
  ),
  file.path(OUT_DIR, "summary_table.tex")
)


# ── Sumário final ─────────────────────────────────────────────────────────────
cat("\n", strrep("=", 60), "\n")
cat("SUMÁRIO\n")
cat(strrep("=", 60), "\n")
cat(sprintf("Cenários analisados  : %d\n", nrow(friedman_mat)))
cat(sprintf("Configs testadas     : %d (2S+2L por algo + GREEDY)\n", n_distinct(raw$algo)))
cat(sprintf("Seeds por config     : %d\n", N_SEEDS))
cat(sprintf("Friedman chi²=%.3f  p=%.6f  sig=%s\n",
            ft$statistic, ft$p.value,
            ifelse(ft$p.value < 0.05, "SIM ✓", "NÃO")))
cat("\nOutputs em", OUT_DIR, ":\n")
for (f in list.files(OUT_DIR, full.names=TRUE)) cat("  ", f, "\n")
