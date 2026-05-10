# api.py

import os
import csv
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Dict, Any

load_dotenv()

app = FastAPI(title="TourismRouteSystem API")

# Store de sessoes em memoria: session_id -> {last_result, original_query}
sessions: Dict[str, Dict[str, Any]] = {}

# -- Health check ------------------------------------------------------
@app.api_route("/health", methods=["GET", "HEAD"])
async def health():
    return {"status": "ok"}

# -- Inicializar sistema (uma vez ao arrancar) -------------------------
from main_system import TourismRouteSystem
from src.transit.transit_service import TransitService

system = None
transit_service = None

@app.on_event("startup")
async def startup():
    global system, transit_service
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY nao configurada")
    system = TourismRouteSystem(api_key=api_key)

    try:
        from src.transit.transit_service import TransitService
        transit_service = TransitService()
        transit_service.load(use_cache=True)
        if transit_service.graph and transit_service.graph.number_of_nodes() > 0:
            print(f"TransitService OK: {transit_service.graph.number_of_nodes()} paragens")
        else:
            print("AVISO: TransitService: grafo vazio - transportes publicos usarao Haversine")
        system.transit_service = transit_service
    except Exception as e:
        print(f"AVISO: TransitService nao disponivel: {e}")

# -- Pastas de output --------------------------------------------------
Path("outputs/maps").mkdir(parents=True, exist_ok=True)
Path("data/feedback").mkdir(parents=True, exist_ok=True)
FEEDBACK_CSV = Path("data/feedback/responses.csv")

# -- Modelos Pydantic --------------------------------------------------
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class FeedbackRequest(BaseModel):
    p1: int; p2: int; p3: int; p4: int; p5: int
    p6: int; p7: int; p8: int; p9: int; p10: int
    p11: int; p12: int; p13: int; p14: int; p15: int
    p16_loc: Optional[int] = 0
    p17_time: Optional[int] = 0
    p18: Optional[str] = ""
    p19: Optional[str] = ""
    p20_age: Optional[str] = ""
    p21_ai: Optional[str] = ""
    p22_travel: Optional[str] = ""
    p23: Optional[str] = ""
    run_id: Optional[str] = None

# -- ENDPOINT 1: GET / - serve index.html -----------------------------
@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = Path("index.html")
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="index.html nao encontrado")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))

# -- Aplica operacao de refinamento sobre a rota existente ------------
def apply_refinement(operation: Dict[str, Any], last_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Modifica a rota existente com base na operacao devolvida pelo LLM.
    Suporta: remove (POI especifico) e filter_category (categoria inteira).
    Devolve novo resultado com a rota modificada.
    """
    route = list(last_result.get("route", []))
    op_type = operation.get("type", "fresh_query")

    if op_type == "remove":
        nomes = [n.lower() for n in operation.get("poi_names", [])]
        route = [p for p in route if p.get("name", "").lower() not in nomes]

    elif op_type == "filter_category":
        excluir = [c.lower() for c in operation.get("exclude_categories", [])]
        route = [p for p in route if p.get("category", "").lower() not in excluir]

    modified = dict(last_result)
    modified["route"] = route
    modified["refinement_applied"] = op_type
    return modified


# -- ENDPOINT 2: POST /query - processa query e devolve rota ----------
@app.post("/query")
async def query_route(req: QueryRequest, request: Request):
    if not system:
        raise HTTPException(status_code=503, detail="Sistema nao inicializado")
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query vazia")

    import time
    t_start = time.time()

    # Detectar refinamento: sessao existente com resultado anterior
    session_id = req.session_id
    is_refinement = session_id and session_id in sessions and sessions[session_id].get("last_result")

    result = None
    effective_query = req.query  # pode ser substituida por query composta abaixo

    if is_refinement:
        last_result = sessions[session_id]["last_result"]
        try:
            operation = system.llm.interpret_refinement(req.query, last_result.get("route", []))
            print(f"   Operacao de refinamento: {operation}")
            if operation.get("type") == "fresh_query":
                # Manter contexto da sessao: juntar query original com a nova instrucao
                base_query = sessions[session_id].get("original_query", "")
                if base_query and base_query.strip() != req.query.strip():
                    effective_query = f"{base_query}. Actualizacao: {req.query}"
                    print(f"   Contexto preservado - query combinada")
                is_refinement = False
            else:
                result = apply_refinement(operation, last_result)
                result["is_refinement"] = True
        except Exception as e:
            print(f"AVISO: Erro no refinamento: {e} - a processar como query nova")
            is_refinement = False

    if not is_refinement:
        try:
            result = system.plan_route(
                effective_query,
                use_shap=False,
                verbose=True,
                force_algorithm=None,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    if result.get("status") == "needs_clarification":
        result["session_id"] = session_id or str(uuid.uuid4())[:8]
        return JSONResponse(content=result)

    # Gerar ID unico para o mapa (apenas em rotas novas)
    if not is_refinement:
        map_id = str(uuid.uuid4())[:8]
        map_path = Path(f"outputs/maps/{map_id}.html")
        if result.get("map_file"):
            original = Path(result["map_file"])
            if original.exists():
                original.rename(map_path)
                result["map_id"] = map_id
            else:
                result["map_id"] = None
        else:
            result["map_id"] = None

    # Limpar campos nao serializaveis antes de devolver
    result.pop("map_file", None)
    result.pop("shap_explanation", None)

    run_id = None
    if not is_refinement:
        try:
            from scripts.log_to_hf import log_run
            client_ip = request.client.host if request.client else None
            run_id = log_run(
                query=req.query,
                result=result,
                elapsed_seconds=time.time() - t_start,
                map_id=result.get("map_id"),
                user_ip=client_ip,
            )
        except Exception:
            pass
    result["run_id"] = run_id

    # Guardar sessao para possiveis refinamentos futuros
    if not session_id:
        session_id = str(uuid.uuid4())[:8]
    sessions[session_id] = {"last_result": dict(result), "original_query": effective_query}
    result["session_id"] = session_id

    return JSONResponse(content=result)

# -- ENDPOINT 3: GET /map/{map_id} - serve mapa HTML ------------------
@app.get("/map/{map_id}", response_class=HTMLResponse)
async def get_map(map_id: str):
    map_path = Path(f"outputs/maps/{map_id}.html")
    if not map_path.exists():
        raise HTTPException(status_code=404, detail="Mapa nao encontrado")
    return HTMLResponse(content=map_path.read_text(encoding="utf-8"))

# -- ENDPOINT 4: POST /feedback - guarda respostas SUS ----------------
@app.post("/feedback")
async def save_feedback(fb: FeedbackRequest):
    # Calcular score SUS
    odd  = (fb.p1-1) + (fb.p3-1) + (fb.p5-1) + (fb.p7-1) + (fb.p9-1)
    even = (5-fb.p2) + (5-fb.p4) + (5-fb.p6) + (5-fb.p8) + (5-fb.p10)
    sus_score = (odd + even) * 2.5

    # Escrever CSV
    write_header = not FEEDBACK_CSV.exists()
    with open(FEEDBACK_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "timestamp",
                "p1","p2","p3","p4","p5","p6","p7","p8","p9","p10",
                "sus_score",
                "p11","p12","p13","p14","p15",
                "p16_loc","p17_time",
                "p18_open","p19_open",
                "p20_age","p21_ai","p22_travel","p23_planeamento"
            ])
        writer.writerow([
            datetime.utcnow().isoformat(),
            fb.p1, fb.p2, fb.p3, fb.p4, fb.p5,
            fb.p6, fb.p7, fb.p8, fb.p9, fb.p10,
            sus_score,
            fb.p11, fb.p12, fb.p13, fb.p14, fb.p15,
            fb.p16_loc, fb.p17_time,
            fb.p18, fb.p19,
            fb.p20_age, fb.p21_ai, fb.p22_travel, fb.p23
        ])

    if fb.run_id:
        try:
            from scripts.log_to_hf import log_feedback
            log_feedback(
                run_id=fb.run_id,
                feedback_data={
                    "p1": fb.p1, "p2": fb.p2, "p3": fb.p3, "p4": fb.p4, "p5": fb.p5,
                    "p6": fb.p6, "p7": fb.p7, "p8": fb.p8, "p9": fb.p9, "p10": fb.p10,
                    "p11": fb.p11, "p12": fb.p12, "p13": fb.p13, "p14": fb.p14, "p15": fb.p15,
                    "p16_loc": fb.p16_loc, "p17_time": fb.p17_time,
                    "p18": fb.p18, "p19": fb.p19,
                    "p20_age": fb.p20_age, "p21_ai": fb.p21_ai,
                    "p22_travel": fb.p22_travel, "p23": fb.p23,
                },
                sus_score=sus_score
            )
        except Exception:
            pass

    return {"status": "ok", "sus_score": sus_score}

# -- ENDPOINT 5: GET /admin - descarregar CSV (password protegido) -----
@app.get("/admin")
async def download_csv(x_admin_password: Optional[str] = Header(None)):
    admin_pw = os.getenv("ADMIN_PASSWORD", "thesis2025")
    if x_admin_password != admin_pw:
        raise HTTPException(status_code=401, detail="Password incorrecta")
    if not FEEDBACK_CSV.exists():
        raise HTTPException(status_code=404, detail="Ainda nao ha respostas")
    return FileResponse(
        path=str(FEEDBACK_CSV),
        media_type="text/csv",
        filename="sus_responses.csv"
    )

# -- ENDPOINT extra: GET /feedback - serve feedback.html --------------
@app.get("/feedback", response_class=HTMLResponse)
async def feedback_page():
    html_path = Path("feedback.html")
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="feedback.html nao encontrado")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))