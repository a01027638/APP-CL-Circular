# CL-Circular-Analytics

Plataforma local de analitica estrategica para identificar clientes potenciales de monitoreo de cadena de frio en el corredor comercial Mexico-Estados Unidos para carnes congeladas bovinas y porcinas.

## Objetivo

Responder:

**Donde estan los clientes potenciales con mayor riesgo logistico y mayor valor economico para CL Circular?**

La plataforma incluye:

1. Mercado y KPIs
2. Ranking de empresas (Opportunity Score configurable)
3. Mapa de flujos MX -> US
4. Simulador de escenarios
5. Pronosticos (ARIMA/SARIMA/Holt-Winters)
6. Clustering de clientes (K-Means + PCA + codo/silhouette)
7. Generador de leads con export CSV

## Estructura

```text
CL-Circular-Analytics
├── app
│   ├── Home.py
│   └── pages
│       ├── 1_Mercado_KPIs.py
│       ├── 2_Ranking_Empresas.py
│       ├── 3_Mapa_Flujos.py
│       ├── 4_Simulador_Escenarios.py
│       ├── 5_Pronosticos.py
│       ├── 6_Clustering.py
│       └── 7_Generador_Leads.py
├── analytics
├── config
├── data
├── outputs
├── requirements.txt
└── README.md
```

## Requisitos

- Python 3.10+
- Visual Studio Code

## Instalacion

```bash
cd CL-Circular-Analytics
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Generar datos mock

```bash
python data/generate_mock_data.py
```

Notas:

- Se generan automaticamente 100 empresas, 200 flujos y 60 meses de historial mensual.
- Si no existen los CSV, la app los crea al iniciar.

## Ejecutar aplicacion

```bash
streamlit run app/Home.py
```

## Frontend HTML + Backend API (para presentacion cliente)

Arquitectura:

- `backend/main.py`: API FastAPI reutilizando modulos de analitica Python.
- `frontend/index.html`: interfaz comercial con estilos modernos.
- `frontend/app.js`: consumo de endpoints, graficas Plotly y mapa Leaflet.

Ejecutar:

```bash
uvicorn backend.main:app --reload --port 8000
```

Abrir en navegador:

```text
http://localhost:8000
```

## Modelo de oportunidad

Opportunity Score:

```text
(volumen_exportado * w1)
+ (valor_mercancia * w2)
+ (distancia_logistica * w3)
+ (sensibilidad_temperatura * w4)
+ (riesgo_congestion * w5)
+ (incidencias_proxy * w6)
```

Los pesos `w1..w6` se ajustan desde la pagina de ranking.

## Salidas

- Visualizaciones interactivas en Streamlit
- Tabla de leads priorizados
- Export CSV desde la pagina `7_Generador_Leads.py`
