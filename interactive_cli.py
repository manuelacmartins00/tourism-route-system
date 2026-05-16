# interactive_cli.py (VERSAO COMPLETA COM ESCOLHA DE ALGORITMO + METRICAS)

import os
import sys
import json
from pathlib import Path
from colorama import init, Fore, Style
import time
import numpy as np

# Inicializar colorama
init(autoreset=True)

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent))

from main_system import TourismRouteSystem
from src.utils.metrics_evaluator import MetricsEvaluator

class InteractiveCLI:
    """Interface de linha de comando interativa com escolha de algoritmo"""
    
    def __init__(self):
        self.system = None
        self.history = []
        self.metrics_evaluator = None
        
    def print_header(self):
        """Cabecalho"""
        print(f"\n{Fore.CYAN}{'='*70}")
        print(f"{Fore.CYAN}SISTEMA DE RECOMENDACAO DE ROTAS TURISTICAS")
        print(f"{Fore.CYAN}{'='*70}\n")
    
    def print_menu(self):
        """Menu principal"""
        print(f"\n{Fore.YELLOW}Opcoes:")
        print(f"{Fore.GREEN}  1. {Fore.WHITE}Planear nova rota (escolher algoritmo)")
        print(f"{Fore.GREEN}  2. {Fore.WHITE}Comparar todos os algoritmos (mesma query)")
        print(f"{Fore.GREEN}  3. {Fore.WHITE}Ver historico de rotas")
        print(f"{Fore.GREEN}  4. {Fore.WHITE}Ver metricas de avaliacao")
        print(f"{Fore.GREEN}  5. {Fore.WHITE}Configuracoes")
        print(f"{Fore.GREEN}  6. {Fore.WHITE}Ajuda")
        print(f"{Fore.GREEN}  7. {Fore.WHITE}Sair")
        print()
    
    def initialize_system(self):
        """Inicializa o sistema"""
        print(f"{Fore.YELLOW}A inicializar sistema...\n")

        # Verificar API key
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print(f"{Fore.RED}[ERRO] GROQ_API_KEY nao configurada!")
            print(f"{Fore.YELLOW}\nPara configurar:")
            print(f"{Fore.WHITE}  1. Vai a https://console.groq.com")
            print(f"{Fore.WHITE}  2. Cria uma API Key")
            print(f"{Fore.WHITE}  3. Adiciona ao .env: GROQ_API_KEY=gsk_...")
            print(f"{Fore.WHITE}  4. Reinicia o terminal\n")
            return False
        
        try:
            self.system = TourismRouteSystem(api_key=api_key)
            self.metrics_evaluator = MetricsEvaluator()
            print(f"{Fore.GREEN}[OK] Sistema pronto!\n")
            return True
        except Exception as e:
            print(f"{Fore.RED}[ERRO] Erro ao inicializar: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def choose_algorithm(self):
        """Menu para escolher algoritmo"""
        print(f"\n{Fore.CYAN}{'-'*70}")
        print(f"{Fore.CYAN}ESCOLHER ALGORITMO DE OTIMIZACAO")
        print(f"{Fore.CYAN}{'-'*70}\n")
        
        print(f"{Fore.YELLOW}Algoritmos disponiveis:")
        print(f"{Fore.GREEN}  1. {Fore.WHITE}ACO (Ant Colony Optimization)")
        print(f"{Fore.WHITE}     +-- Melhor para: problemas multi-objetivo, boa exploracao")
        print(f"{Fore.WHITE}     +-- Tempo: ~10-15s | Qualidade: Alta")
        print()
        print(f"{Fore.GREEN}  2. {Fore.WHITE}GA (Genetic Algorithm)")
        print(f"{Fore.WHITE}     +-- Melhor para: grandes espacos de busca, diversidade")
        print(f"{Fore.WHITE}     +-- Tempo: ~8-12s | Qualidade: Alta")
        print()
        print(f"{Fore.GREEN}  3. {Fore.WHITE}PSO (Particle Swarm Optimization)")
        print(f"{Fore.WHITE}     +-- Melhor para: convergencia rapida, otimizacao continua")
        print(f"{Fore.WHITE}     +-- Tempo: ~10-15s | Qualidade: Media-Alta")
        print()
        print(f"{Fore.GREEN}  4. {Fore.WHITE}GREEDY (Algoritmo Guloso)")
        print(f"{Fore.WHITE}     +-- Melhor para: baseline rapido, solucoes simples")
        print(f"{Fore.WHITE}     +-- Tempo: <1s | Qualidade: Media")
        print()
        print(f"{Fore.GREEN}  5. {Fore.WHITE}AUTO (Deixar o LLM escolher)")
        print(f"{Fore.WHITE}     +-- LLM analisa a query e escolhe o melhor")
        print()
        
        while True:
            choice = input(f"{Fore.GREEN}Escolhe algoritmo (1-5): {Fore.WHITE}").strip()
            
            if choice == '1':
                return "ACO"
            elif choice == '2':
                return "GA"
            elif choice == '3':
                return "PSO"
            elif choice == '4':
                return "GREEDY"
            elif choice == '5':
                return None  # AUTO
            else:
                print(f"{Fore.RED}[ERRO] Opcao invalida! Escolhe 1-5.")
    
    def plan_route(self):
        """Planear nova rota com escolha de algoritmo"""
        print(f"\n{Fore.CYAN}{'-'*70}")
        print(f"{Fore.CYAN}PLANEAR NOVA ROTA")
        print(f"{Fore.CYAN}{'-'*70}\n")

        # Exemplos
        print(f"{Fore.YELLOW}Exemplos de queries:")
        print(f"{Fore.WHITE}  - \"quero visitar museus e monumentos, tenho 5 horas e 40 euros\"")
        print(f"{Fore.WHITE}  - \"procuro restaurantes bons e miradouros, 3 horas, 50 euros\"")
        print(f"{Fore.WHITE}  - \"I want to see historic sites and eat well, 6 hours, 60 euros\"")
        print()

        # Input do utilizador
        query = input(f"{Fore.GREEN}A tua query: {Fore.WHITE}").strip()

        if not query:
            print(f"{Fore.RED}[ERRO] Query vazia!")
            return

        # Escolher algoritmo
        force_algorithm = self.choose_algorithm()

        if force_algorithm:
            print(f"\n{Fore.YELLOW}OK Algoritmo selecionado: {Fore.GREEN}{force_algorithm}")
        else:
            print(f"\n{Fore.YELLOW}OK Modo AUTO - LLM vai escolher o algoritmo")

        print(f"\n{Fore.YELLOW}A processar (pode demorar 10-30 segundos)...\n")

        # Estado das respostas de scope (None = ainda nao respondido)
        include_accommodation = None
        include_meals = None

        t_start = time.time()

        try:
            # --- Passagem 1: extrair preferencias e detetar campos em falta + scope ---
            result = self.system.plan_route(
                query,
                use_shap=True,
                verbose=True,
                force_algorithm=force_algorithm,
                include_accommodation=include_accommodation,
                include_meals=include_meals,
            )

            # -- Campos obrigatorios em falta (location, max_time, etc.) --
            if result.get("status") == "needs_clarification":
                print(f"\n{Fore.YELLOW}Precisamos de mais informacao para planear a rota:\n")
                field_questions = {
                    "location":       "Qual a localizacao / cidade?",
                    "max_time":       "Qual a duracao da viagem? (ex: 2 dias, 5 horas)",
                    "max_cost":       "Qual o orcamento disponivel? (ex: 50 euros por pessoa)",
                    "budget_type":    "O orcamento e por pessoa, por dia ou total?",
                    "transport_mode": "Como se vai deslocar? (a pe / carro / transportes publicos)",
                }
                clarifications = []
                for field in result.get("missing_fields", []):
                    q = field_questions.get(field, f"Podes especificar '{field}'?")
                    ans = input(f"{Fore.GREEN}  {q} {Fore.WHITE}").strip()
                    if ans:
                        clarifications.append(ans)

                if not clarifications:
                    print(f"{Fore.RED}Sem informacao adicional — nao e possivel planear a rota.")
                    return

                query = query + ". " + ". ".join(clarifications)
                print(f"\n{Fore.YELLOW}A reprocessar...\n")
                result = self.system.plan_route(
                    query,
                    use_shap=True,
                    verbose=True,
                    force_algorithm=force_algorithm,
                    include_accommodation=include_accommodation,
                    include_meals=include_meals,
                )

            if result.get("status") == "needs_clarification":
                print(f"{Fore.RED}[ERRO] Ainda faltam campos: {result.get('missing_fields')}")
                return

            # -- Perguntas de scope: alojamento e/ou refeicoes --
            if result.get("status") == "needs_scope_clarification":
                scope_qs = result.get("scope_questions", [])
                prefs = result.get("preferences_so_far", {})

                print(f"\n{Fore.CYAN}{'-'*70}")
                print(f"{Fore.CYAN}ALOJAMENTO E REFEICOES")
                print(f"{Fore.CYAN}{'-'*70}")
                print(f"{Fore.YELLOW}Nota: {Fore.WHITE}Muitos utilizadores preferem escolher alojamento e refeicoes")
                print(f"{Fore.WHITE}autonomamente via Booking, Airbnb, Google Maps, TripAdvisor, etc.")
                print()

                if "include_accommodation" in scope_qs:
                    days = max(1, prefs.get("max_time", 480) // 480)
                    ans = input(
                        f"{Fore.GREEN}A tua viagem tem {days}+ dia(s). Incluir sugestoes de "
                        f"{Fore.CYAN}alojamento{Fore.GREEN} na rota? (s/n): {Fore.WHITE}"
                    ).strip().lower()
                    include_accommodation = ans in ("s", "sim", "y", "yes", "1")
                else:
                    include_accommodation = True

                if "include_meals" in scope_qs:
                    ans = input(
                        f"{Fore.GREEN}A rota inclui horas de refeicao. Incluir sugestoes de "
                        f"{Fore.CYAN}restaurantes/cafes{Fore.GREEN} na rota? (s/n): {Fore.WHITE}"
                    ).strip().lower()
                    include_meals = ans in ("s", "sim", "y", "yes", "1")
                else:
                    include_meals = True

                print()
                if not include_accommodation:
                    print(f"{Fore.YELLOW}  Alojamento: {Fore.WHITE}tratas autonomamente (Booking/Airbnb)")
                if not include_meals:
                    print(f"{Fore.YELLOW}  Refeicoes:  {Fore.WHITE}tratas autonomamente (Google Maps/TripAdvisor)")
                print()

                print(f"{Fore.YELLOW}A gerar a rota final...\n")
                result = self.system.plan_route(
                    query,
                    use_shap=True,
                    verbose=True,
                    force_algorithm=force_algorithm,
                    include_accommodation=include_accommodation,
                    include_meals=include_meals,
                )

            if result.get("status") in ("needs_clarification", "needs_scope_clarification"):
                print(f"{Fore.RED}[ERRO] Nao foi possivel completar o planeamento.")
                return

            elapsed = time.time() - t_start

            # Calcular metricas
            metrics = self.metrics_evaluator.calculate_metrics(result)

            # Guardar no historico
            self.history.append({
                'query': query,
                'result': result,
                'metrics': metrics,
                'algorithm': result.get('algorithm_used', '-'),
                'forced': force_algorithm is not None,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'elapsed_seconds': elapsed
            })

            # Mostrar metricas
            self._display_metrics(metrics, result.get('algorithm_used', '-'))

            # Perguntar se quer guardar
            save = input(f"\n{Fore.YELLOW}Guardar resultado em ficheiro? (s/n): {Fore.WHITE}").strip().lower()
            if save == 's':
                self.save_result(result, metrics)

        except Exception as e:
            print(f"{Fore.RED}[ERRO] Erro ao processar: {e}")
            import traceback
            traceback.print_exc()
    
    def compare_algorithms(self):
        """Comparar todos os algoritmos com a mesma query"""
        print(f"\n{Fore.CYAN}{'-'*70}")
        print(f"{Fore.CYAN}COMPARAR TODOS OS ALGORITMOS")
        print(f"{Fore.CYAN}{'-'*70}\n")
        
        print(f"{Fore.YELLOW}Esta opcao vai executar ACO, GA, PSO e GREEDY com a mesma query.")
        print(f"{Fore.YELLOW}Pode demorar 1-2 minutos.\n")
        
        # Input
        query = input(f"{Fore.GREEN}Query para comparacao: {Fore.WHITE}").strip()

        if not query:
            print(f"{Fore.RED}[ERRO] Query vazia!")
            return

        algorithms = ["ACO", "GA", "PSO", "GREEDY"]
        results = {}
        metrics_all = {}

        print(f"\n{Fore.YELLOW}A executar comparacao...\n")
        
        for algo in algorithms:
            print(f"{Fore.CYAN}{'-'*70}")
            print(f"{Fore.CYAN}Testando {algo}...")
            print(f"{Fore.CYAN}{'-'*70}\n")
            
            start_time = time.time()
            
            try:
                result = self.system.plan_route(
                    query,
                    use_shap=False,  # Desativar SHAP para ser mais rapido
                    verbose=False,
                    force_algorithm=algo,
                    include_accommodation=True,
                    include_meals=True,
                )

                if result.get("status") in ("needs_clarification", "needs_scope_clarification"):
                    print(f"{Fore.RED}X {algo} — query incompleta: {result.get('missing_fields') or result.get('scope_questions')}")
                    results[algo] = None
                    metrics_all[algo] = None
                    continue

                elapsed = time.time() - start_time

                # Calcular metricas
                metrics = self.metrics_evaluator.calculate_metrics(result)

                results[algo] = result
                metrics_all[algo] = metrics

                print(f"{Fore.GREEN}OK {algo} completado em {elapsed:.1f}s")
                print(f"  Fitness: {result['optimization']['fitness']:.2f}")
                print(f"  POIs: {len(result['route'])}")
            
            except Exception as e:
                print(f"{Fore.RED}X {algo} falhou: {e}\n")
                results[algo] = None
                metrics_all[algo] = None
        
        # Comparacao final
        self._display_comparison(results, metrics_all)
        
        # Guardar historico
        self.history.append({
            'type': 'comparison',
            'query': query,
            'results': results,
            'metrics': metrics_all,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Guardar ficheiro
        save = input(f"\n{Fore.YELLOW}Guardar comparacao em ficheiro? (s/n): {Fore.WHITE}").strip().lower()
        
        if save == 's':
            self.save_comparison(query, results, metrics_all)
    
    def _display_comparison(self, results, metrics_all):
        """Mostra tabela de comparacao"""
        print(f"\n{Fore.CYAN}{'='*70}")
        print(f"{Fore.CYAN}COMPARACAO FINAL")
        print(f"{Fore.CYAN}{'='*70}\n")
        
        # Header
        print(f"{Fore.YELLOW}{'Algoritmo':<12} | {'Fitness':>8} | {'POIs':>5} | {'Tempo':>8}")
        print(f"{Fore.YELLOW}{'-'*70}")
        
        # Resultados
        for algo in ["ACO", "GA", "PSO", "GREEDY"]:
            if results[algo] is None:
                print(f"{Fore.RED}{algo:<12} | {'ERRO':>8} | {'-':>5} | {'-':>8} | {'-':>8} | {'-':>8}")
                continue
            
            result = results[algo]
            metrics = metrics_all[algo]
            
            fitness = result['optimization']['fitness']
            n_pois = len(result['route'])
            
            # Colorir melhor resultado
            color = Fore.WHITE
            if fitness == max(r['optimization']['fitness'] for r in results.values() if r):
                color = Fore.GREEN
            
            print(f"{color}{algo:<12} | {fitness:8.2f} | {n_pois:5d} | {'-':>8}")
        
        # Determinar melhor
        best_algo = max(
            [(algo, r) for algo, r in results.items() if r],
            key=lambda x: x[1]['optimization']['fitness']
        )
        
        print(f"\n{Fore.GREEN}Melhor algoritmo: {best_algo[0]} (Fitness: {best_algo[1]['optimization']['fitness']:.2f})")

        # Metricas agregadas
        print(f"\n{Fore.YELLOW}Estatisticas:")
        valid_metrics = [m for m in metrics_all.values() if m]
        
        if valid_metrics:
            std_fitness = np.std([results[a]['optimization']['fitness'] for a in results if results[a]])
            
            print(f"{Fore.WHITE}  Desvio padrao fitness: {std_fitness:.2f}")
    
    def _display_metrics(self, metrics, algorithm):
        """Mostra metricas de um resultado"""
        print(f"\n{Fore.CYAN}{'='*70}")
        print(f"{Fore.CYAN}METRICAS DE AVALIACAO - {algorithm}")
        print(f"{Fore.CYAN}{'='*70}\n")

        print(f"\n{Fore.YELLOW}Qualidade da Rota:")
        print(f"{Fore.WHITE}  Fitness Score: {metrics['fitness_score']:.2f}")
        print(f"{Fore.WHITE}  Coverage (POIs vs candidatos): {metrics['coverage']:.2f}%")
        print(f"{Fore.WHITE}  Constraint Satisfaction: {metrics['constraint_satisfaction']:.2f}%")
        
        print(f"\n{Fore.YELLOW}Eficiencia:")
        print(f"{Fore.WHITE}  Tempo Total: {metrics['total_time']:.0f} min")
        print(f"{Fore.WHITE}  Custo Total: EUR{metrics['total_cost']:.2f}")
        print(f"{Fore.WHITE}  POIs por Euro: {metrics['pois_per_euro']:.2f}")
        print(f"{Fore.WHITE}  POIs por Hora: {metrics['pois_per_hour']:.2f}")
        
        print(f"\n{Fore.YELLOW}Diversidade:")
        print(f"{Fore.WHITE}  Categorias Unicas: {metrics['unique_categories']}")
        print(f"{Fore.WHITE}  Indice de Diversidade: {metrics['diversity_index']:.2f}")
    
    def save_result(self, result, metrics):
        """Guarda resultado individual"""
        Path("outputs").mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        algo = result['algorithm_used']
        filename = f"outputs/route_{algo}_{timestamp}.json"
        
        output = {
            'result': result,
            'metrics': metrics
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"{Fore.GREEN}OK Guardado em: {filename}")
    
    def save_comparison(self, query, results, metrics_all):
        """Guarda comparacao"""
        Path("outputs/comparisons").mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"outputs/comparisons/comparison_{timestamp}.json"
        
        output = {
            'query': query,
            'timestamp': timestamp,
            'results': results,
            'metrics': metrics_all
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"{Fore.GREEN}OK Comparacao guardada em: {filename}")
    
    def show_history(self):
        """Mostra historico"""
        print(f"\n{Fore.CYAN}{'-'*70}")
        print(f"{Fore.CYAN}HISTORICO DE ROTAS")
        print(f"{Fore.CYAN}{'-'*70}\n")
        
        if not self.history:
            print(f"{Fore.YELLOW}Ainda nao planeaste nenhuma rota nesta sessao.")
            return
        
        for i, entry in enumerate(self.history, 1):
            if entry.get('type') == 'comparison':
                print(f"{Fore.GREEN}{i}. {Fore.WHITE}[{entry['timestamp']}] COMPARACAO")
                print(f"   Query: {entry['query']}")
                valid = sum(1 for r in entry['results'].values() if r)
                print(f"   Algoritmos testados: {valid}/4")
            else:
                print(f"{Fore.GREEN}{i}. {Fore.WHITE}[{entry['timestamp']}]")
                print(f"   Query: {entry['query']}")
                print(f"   Algoritmo: {entry['algorithm']}")
                print(f"   POIs: {len(entry['result']['route'])}")
                print(f"   Tempo: {entry['elapsed_seconds']:.1f}s")
            print()
    
    def show_metrics_summary(self):
        """Mostra resumo de metricas do historico"""
        print(f"\n{Fore.CYAN}{'-'*70}")
        print(f"{Fore.CYAN}RESUMO DE METRICAS")
        print(f"{Fore.CYAN}{'-'*70}\n")
        
        if not self.history:
            print(f"{Fore.YELLOW}Sem dados no historico.")
            return
        
        # Filtrar apenas rotas individuais
        routes = [h for h in self.history if h.get('type') != 'comparison']
        
        if not routes:
            print(f"{Fore.YELLOW}Sem rotas individuais no historico.")
            return
        
        # Agrupar por algoritmo
        by_algo = {}
        for route in routes:
            algo = route['algorithm']
            if algo not in by_algo:
                by_algo[algo] = []
            by_algo[algo].append(route['metrics'])

        # Estatisticas por algoritmo
        print(f"{Fore.YELLOW}Estatisticas por Algoritmo:\n")
        
        for algo, metrics_list in by_algo.items():
            print(f"{Fore.GREEN}{algo}:")
            
            avg_fitness = np.mean([m['fitness_score'] for m in metrics_list])
            
            print(f"{Fore.WHITE}  Execucoes: {len(metrics_list)}")
            print(f"{Fore.WHITE}  Fitness medio: {avg_fitness:.2f}")
            print()
    
    def show_settings(self):
        """Mostra configuracoes"""
        print(f"\n{Fore.CYAN}{'-'*70}")
        print(f"{Fore.CYAN}CONFIGURACOES")
        print(f"{Fore.CYAN}{'-'*70}\n")
        
        api_key = os.getenv("HF_TOKEN")
        
        print(f"{Fore.YELLOW}API Key:")
        if api_key:
            print(f"{Fore.GREEN}   [OK] Configurada: {api_key[:20]}...{api_key[-10:]}")
        else:
            print(f"{Fore.RED}   [N/A] Nao configurada")
        
        print(f"\n{Fore.YELLOW}Modelo LLM:")
        print(f"{Fore.WHITE}   llama-3.1-8b-instant (Groq)")
        
        print(f"\n{Fore.YELLOW}Algoritmos disponiveis:")
        print(f"{Fore.WHITE}   - ACO (Ant Colony Optimization)")
        print(f"{Fore.WHITE}   - GA (Genetic Algorithm)")
        print(f"{Fore.WHITE}   - PSO (Particle Swarm Optimization)")
        print(f"{Fore.WHITE}   - GREEDY (Baseline)")

        print(f"\n{Fore.YELLOW}Base de dados:")
        print(f"{Fore.WHITE}   - ChromaDB (RAG)")
        print(f"{Fore.WHITE}   - 25 POIs em Lisboa")

        print(f"\n{Fore.YELLOW}Metricas calculadas:")
        print(f"{Fore.WHITE}   - Coverage, Diversidade, Eficiencia")
    
    def show_help(self):
        """Ajuda"""
        print(f"\n{Fore.CYAN}{'-'*70}")
        print(f"{Fore.CYAN}AJUDA")
        print(f"{Fore.CYAN}{'-'*70}\n")
        
        print(f"{Fore.YELLOW}Como usar:")
        print(f"{Fore.WHITE}  1. Escreve a tua query em linguagem natural (PT ou EN)")
        print(f"{Fore.WHITE}  2. Escolhe o algoritmo (ACO/GA/PSO/GREEDY) ou deixa AUTO")
        print(f"{Fore.WHITE}  3. O sistema processa e mostra resultado + metricas")
        print(f"{Fore.WHITE}  4. Opcionalmente, compara todos os algoritmos de uma vez")
        
        print(f"\n{Fore.YELLOW}Algoritmos:")
        print(f"{Fore.WHITE}  - ACO: Otimizacao inspirada em formigas, boa exploracao")
        print(f"{Fore.WHITE}  - GA: Algoritmo genetico, bom para espacos grandes")
        print(f"{Fore.WHITE}  - PSO: Enxame de particulas, convergencia rapida")
        print(f"{Fore.WHITE}  - GREEDY: Baseline simples e rapido")
        
    
    def run(self):
        """Loop principal"""
        self.print_header()
        
        if not self.initialize_system():
            return
        
        while True:
            self.print_menu()
            
            choice = input(f"{Fore.GREEN}Escolhe (1-7): {Fore.WHITE}").strip()
            
            if choice == '1':
                self.plan_route()
            elif choice == '2':
                self.compare_algorithms()
            elif choice == '3':
                self.show_history()
            elif choice == '4':
                self.show_metrics_summary()
            elif choice == '5':
                self.show_settings()
            elif choice == '6':
                self.show_help()
            elif choice == '7':
                print(f"\n{Fore.CYAN}Ate breve!")
                break
            else:
                print(f"{Fore.RED}[ERRO] Opcao invalida!")

if __name__ == "__main__":
    cli = InteractiveCLI()
    cli.run()