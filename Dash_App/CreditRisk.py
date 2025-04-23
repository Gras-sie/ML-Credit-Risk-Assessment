import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import pickle
import os
import warnings
import json

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    external_scripts=[
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/js/all.min.js"
    ],
    suppress_callback_exceptions=True
)
app.title = "Credit Risk Predictor"

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

model_paths = {
    "random_forest": os.path.join(project_root, "Artifacts", "PLK", "random_forest_model.pkl"),
    "xgboost": os.path.join(project_root, "Artifacts", "PLK", "xgboost_model.json")  
}
label_map_path = os.path.join(project_root, "Artifacts", "PLK", "label_mapping.json")
columns_path = os.path.join(project_root, "Artifacts", "PLK", "training_columns.pkl")
data_file_path = os.path.join(project_root, "SRC", "credit_risk_features.csv")

try:
    import xgboost
except ImportError:
    print("⚠️ xgboost is not installed. The XGBoost model will be skipped.")
    xgboost = None

from sklearn import __version__ as sklearn_version
expected_version = "1.5.1"
if sklearn_version != expected_version:
    warnings.warn(
        f"⚠️ scikit-learn version mismatch: expected {expected_version}, found {sklearn_version}. "
        "This could lead to compatibility issues when loading models."
    )
try:
    models = {}
    
    with open(label_map_path) as f:
        label_mapping = json.load(f)
        
    for key, path in model_paths.items():
        if key == "xgboost":
            if xgboost is None:
                print("Skipping XGBoost model — library not installed.")
                continue
            if not os.path.exists(path):
                raise FileNotFoundError(f"XGBoost model file not found: {path}")
            try:
                model = xgboost.XGBClassifier(
                    n_estimators=500,
                    eval_metric='mlogloss',
                    objective='multi:softmax',
                    num_class=len(label_mapping),
                    enable_categorical=False
                )
                model.load_model(path)
                models[key] = model
            except Exception as e:
                print(f"Error loading XGBoost model: {e}")
                continue
        else:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")
            with open(path, "rb") as f:
                models[key] = pickle.load(f)

    if not os.path.exists(columns_path):
        raise FileNotFoundError(f"Feature column file not found: {columns_path}")
    with open(columns_path, "rb") as f:
        training_columns = pickle.load(f)

    if not os.path.exists(data_file_path):
        raise FileNotFoundError(f"Data file not found: {data_file_path}")
    full_data = pd.read_csv(data_file_path)

except FileNotFoundError as e:
    print(f"🚫 Error: {e}")
    print("Make sure all model and data files are correctly placed:")
    print(f" - Models → {os.path.dirname(model_paths['random_forest'])}")
    print(f" - Data   → {os.path.dirname(data_file_path)}")
    raise

def flag_to_risk(flag):
    return {
        "P1": "Low Risk - Likely Approved",
        "P2": "Medium Risk - Review Needed",
        "P3": "High Risk - Likely Denied",
        "P4": "Critical Risk - Must Deny"
    }.get(flag, "Unknown")


@app.callback(
    [Output("prediction-output", "children"),
     Output("prediction-explanation", "children")],
    [Input("predict-button", "n_clicks")],
    [State("model-selection", "value"),
     State("input-age", "value"),
     State("input-gender", "value"),
     State("input-marital", "value"),
     State("input-education", "value"),
     State("input-income", "value"),
     State("input-employment-time", "value"),
     State("input-cc-flag", "value"),
     State("input-pl-flag", "value"),
     State("input-hl-flag", "value"),
     State("input-gl-flag", "value"),
     State("input-credit-score", "value"),
     State("input-first-prod-enq", "value"),
     State("input-last-prod-enq", "value")]
)
def predict_approval(n_clicks, model_type, age, gender, marital, education, 
                     income, employment_time, cc_flag, pl_flag, hl_flag, gl_flag,
                     credit_score, first_prod_enq, last_prod_enq):
    if not n_clicks:
        return "", ""

    try:
        numeric_fields = {
            'Age': age,
            'Income': income,
            'Employment Time': employment_time,
            'Credit Score': credit_score
        }
        
        for field, value in numeric_fields.items():
            try:
                if value is not None:
                    float(value) 
            except ValueError:
                return html.Div(
                    f"Invalid value for {field}. Please enter a valid number.", 
                    className="text-danger"
                ), ""

        if model_type not in models:
            model_type = 'random_forest'
        model = models[model_type]

        try:
            base_row = full_data.sample(n=1, random_state=42).copy()
        except Exception as e:
            print(f"Error sampling data: {str(e)}")
            return html.Div(
                "Error preparing input data. Please try again.", 
                className="text-danger"
            ), ""

        input_mappings = {
            'AGE': float(age) if age is not None else None,
            'GENDER': gender,
            'MARITALSTATUS': marital,
            'EDUCATION': education,
            'NETMONTHLYINCOME': float(income) if income is not None else None,
            'Time_With_Curr_Empr': float(employment_time) if employment_time is not None else None,
            'CC_Flag': 1 if cc_flag == 'Yes' else 0,
            'PL_Flag': 1 if pl_flag == 'Yes' else 0,
            'HL_Flag': 1 if hl_flag == 'Yes' else 0,
            'GL_Flag': 1 if gl_flag == 'Yes' else 0,
            'Credit_Score': float(credit_score) if credit_score is not None else None,
            'first_prod_enq2': first_prod_enq or 'Personal Loan',
            'last_prod_enq2': last_prod_enq or 'Credit Card'
        }

        missing_fields = [k for k, v in input_mappings.items() if v is None]
        if missing_fields:
            return html.Div(
                f"Missing required fields: {', '.join(missing_fields)}", 
                className="text-danger"
            ), ""

        for col, val in input_mappings.items():
            if col in base_row.columns:
                base_row[col] = val

        try:
            base_row = base_row.drop(columns=['Approved_Flag'], errors='ignore')
            input_encoded = pd.get_dummies(base_row)
            input_encoded = input_encoded.reindex(columns=training_columns, fill_value=0)
            
            # Convert to float32 to match training data type
            input_encoded = input_encoded.astype('float32')
            
            if model_type == 'xgboost':
                # Use predict method directly
                prediction_raw = models[model_type].predict(input_encoded)[0]
            else:
                prediction_raw = models[model_type].predict(input_encoded)[0]
            
            if isinstance(prediction_raw, (int, np.integer)):
                label_map = {0: "P1", 1: "P2", 2: "P3", 3: "P4"}
                prediction = label_map.get(prediction_raw, str(prediction_raw))
            else:
                prediction = str(prediction_raw)
                
            risk = flag_to_risk(prediction)
            
            return html.H3(
                f"Risk Assessment: {risk}", 
                className="text-info"
            ), html.P(
                f"Based on your application, the {model_type.replace('_', ' ').title()} model classified you as: {risk}. "
                f"This classification considers multiple financial and credit-based indicators."
            )

        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return html.Div(
                "Error during prediction processing. Please verify your inputs and try again.", 
                className="text-danger"
            ), ""

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return html.Div(
            "An unexpected error occurred. Please try again later.", 
            className="text-danger"
        ), ""

@app.callback(
    [Output("comparison-output", "children"),
     Output("tips-output", "children")],
    [Input("compare-button", "n_clicks")],
    [State("input-age", "value"),
     State("input-gender", "value"),
     State("input-marital", "value"),
     State("input-education", "value"),
     State("input-income", "value"),
     State("input-employment-time", "value"),
     State("input-cc-flag", "value"),
     State("input-pl-flag", "value"),
     State("input-hl-flag", "value"),
     State("input-gl-flag", "value"),
     State("input-credit-score", "value"),
     State("input-first-prod-enq", "value"),
     State("input-last-prod-enq", "value")]
)
def compare_models(n_clicks, age, gender, marital, education, 
                  income, employment_time, cc_flag, pl_flag, hl_flag, gl_flag,
                  credit_score, first_prod_enq, last_prod_enq):
    if not n_clicks or None in [age, gender, marital, education, income, employment_time, 
                                cc_flag, pl_flag, hl_flag, gl_flag, credit_score]:
        return html.Div("Please fill in all fields", className="text-danger"), ""
    results = []
    for model_type in ['random_forest', 'xgboost']:
        model = models[model_type]
        base_row = full_data.sample(n=1, random_state=42).copy()
        input_mappings = {
            'AGE': age,
            'GENDER': gender,
            'MARITALSTATUS': marital,
            'EDUCATION': education,
            'NETMONTHLYINCOME': income,
            'Time_With_Curr_Empr': employment_time,
            'CC_Flag': 1 if cc_flag == 'Yes' else 0,
            'PL_Flag': 1 if pl_flag == 'Yes' else 0,
            'HL_Flag': 1 if hl_flag == 'Yes' else 0,
            'GL_Flag': 1 if gl_flag == 'Yes' else 0,
            'Credit_Score': credit_score,
            'first_prod_enq2': first_prod_enq or 'Personal Loan',
            'last_prod_enq2': last_prod_enq or 'Credit Card'
        }
        for col, val in input_mappings.items():
            if col in base_row.columns:
                base_row[col] = val
        base_row = base_row.drop(columns=['Approved_Flag'], errors='ignore')
        input_encoded = pd.get_dummies(base_row)
        input_encoded = input_encoded.reindex(columns=training_columns, fill_value=0)
        prediction_raw = model.predict(input_encoded)[0]
        label_map = {0: "P1", 1: "P2", 2: "P3", 3: "P4"}
        prediction = label_map.get(prediction_raw, prediction_raw)
        risk = flag_to_risk(prediction)
        results.append((model_type.replace('_', ' ').title(), risk))
    comparison_table = dbc.Table([
        html.Thead(html.Tr([html.Th("Model"), html.Th("Risk Assessment")])),
        html.Tbody([
            html.Tr([html.Td(model), html.Td(risk)]) for model, risk in results
        ])
    ], bordered=True, hover=True, responsive=True, className="mt-3")
    tips = html.Ul([
        html.Li("Maintain a good credit score by paying bills on time."),
        html.Li("Keep your credit utilization low."),
        html.Li("Avoid applying for too many loans/credit cards at once."),
        html.Li("Increase your income and job stability where possible."),
        html.Li("Regularly review your credit report for errors.")
    ])
    return comparison_table, tips

app.layout = dbc.Container([
    html.Div(className="particles"),
    html.Div(
        className="card-svg-decoration",
        style={
            "position": "absolute",
            "top": "20px",
            "right": "20px",
            "opacity": "0.1",
            "pointerEvents": "none",
            "zIndex": "-1"
        } 
    ),

    dbc.Navbar([
        html.Img(src="/assets/images/logo.png", height="30px", className="me-2"),
        dbc.NavbarBrand("Credit Risk Predictor", className="ms-2"),
        dbc.Nav(
            [
                dbc.NavLink("Predict", id="nav-predict", href="#predict", 
                           active=True, className="nav-link-current"),
                dbc.NavLink("Advice", id="nav-compare", href="#compare"),  
                dbc.NavLink("About", id="nav-about", href="#about"),
            ],
            className="ms-auto",
            navbar=True,
        )
    ], color="primary", dark=True, sticky="top"),

    dcc.Loading(
        id="global-loading",
        type="circle",
        fullscreen=True,
        children=html.Div(id="loading-placeholder") 
    ),
    
    html.Header([
        html.Div([
            html.H1("Credit Risk Predictor", className="display-4"),
            html.P("Estimate credit approval potential based on client details", className="lead"),
        ], className="header text-center py-4", **{"aria-label": "Application Header"})
    ]),

    html.Main([
        html.Div(id="page-content", role="main", **{"aria-live": "polite"})
    ]),
    html.Footer(
        "© 2024 Credit Risk Predictor | All rights reserved",
        className="text-center py-3 mt-5 border-top"
    )
], fluid=True, className="app-container px-0")

@app.callback(
    [Output("page-content", "children"),
     Output("nav-predict", "active"),
     Output("nav-compare", "active"),
     Output("nav-about", "active")],
    [Input("nav-predict", "n_clicks"),
     Input("nav-compare", "n_clicks"),
     Input("nav-about", "n_clicks")],
    prevent_initial_call=False 
)
def toggle_active_nav(predict_clicks, compare_clicks, about_clicks):
    ctx = dash.callback_context
    
    if not ctx.triggered:
        return render_predict_page(), True, False, False
    
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if triggered_id == "nav-about":
        return render_about_page(), False, False, True
    elif triggered_id == "nav-compare":
        return render_compare_page(), False, True, False
    else: 
        return render_predict_page(), True, False, False

@app.callback(
    [Output("nav-predict", "className"),
     Output("nav-compare", "className"),
     Output("nav-about", "className")],
    [Input("nav-predict", "n_clicks"),
     Input("nav-compare", "n_clicks"),
     Input("nav-about", "n_clicks")]
)
def update_tab_classes(predict_clicks, compare_clicks, about_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return "nav-link-current", "nav-link", "nav-link"

    clicked_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if clicked_id == "nav-predict":
        return "nav-link-current", "nav-link", "nav-link"
    elif clicked_id == "nav-compare":
        return "nav-link", "nav-link-current", "nav-link"
    elif clicked_id == "nav-about":
        return "nav-link", "nav-link", "nav-link-current"

    return "nav-link-current", "nav-link", "nav-link"

def render_predict_page():
    return html.Section([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Applicant Information", className="card-title mb-4"),
                        
                        html.Div([
                            html.H5([
                                html.I(className="fas fa-user me-2"),
                                "Step 1: Personal Details"
                            ]),
                            html.Hr(),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Age", className="form-label"),
                                    dcc.Input(id="input-age", type="number", min=18, max=100, placeholder="Enter age", className="form-control")
                                ]),
                                dbc.Col([
                                    html.Label("Gender", className="form-label"),
                                    dcc.Dropdown(id="input-gender", options=[
                                        {"label": "Male", "value": "Male"},
                                        {"label": "Female", "value": "Female"},
                                        {"label": "Other", "value": "Other"}
                                    ], placeholder="Select gender", className="form-control")
                                ]),
                                dbc.Col([
                                    html.Label("Marital Status", className="form-label"),
                                    dcc.Dropdown(id="input-marital", options=[
                                        {"label": m, "value": m} for m in ["Single", "Married", "Divorced", "Widowed"]
                                    ], placeholder="Marital Status", className="form-control")
                                ]),
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Education", className="form-label"),
                                    dcc.Dropdown(id="input-education", options=[
                                        {"label": edu, "value": edu} for edu in ["High School", "Diploma", "Bachelor's", "Master's", "PhD", "Other"]
                                    ], placeholder="Education Level", className="form-control")
                                ])
                            ])
                        ], className="form-step"),

                        html.Div([
                            html.H5([
                                html.I(className="fas fa-briefcase me-2"),
                                "Step 2: Employment & Income"
                            ]),
                            html.Hr(),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Net Monthly Income (ZAR)", className="form-label"),
                                    dcc.Input(id="input-income", type="number", placeholder="e.g. 10000", className="form-control")
                                ]),
                                dbc.Col([
                                    html.Label("Time With Current Employer (Months)", className="form-label"),
                                    dcc.Input(id="input-employment-time", type="number", placeholder="e.g. 24", className="form-control")
                                ])
                            ])
                        ], className="form-step"),

                        html.Div([
                            html.H5([
                                html.I(className="fas fa-credit-card me-2"),
                                "Step 3: Loan Ownership"
                            ]),
                            html.Hr(),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Has Credit Card?", className="form-label"),
                                    dcc.Dropdown(id="input-cc-flag", options=[
                                        {"label": "Yes", "value": "Yes"},
                                        {"label": "No", "value": "No"}
                                    ], placeholder="Select option", className="form-control")
                                ]),
                                dbc.Col([
                                    html.Label("Has Personal Loan?", className="form-label"),
                                    dcc.Dropdown(id="input-pl-flag", options=[
                                        {"label": "Yes", "value": "Yes"},
                                        {"label": "No", "value": "No"}
                                    ], placeholder="Select option", className="form-control")
                                ]),
                                dbc.Col([
                                    html.Label("Has Home Loan?", className="form-label"),
                                    dcc.Dropdown(id="input-hl-flag", options=[
                                        {"label": "Yes", "value": "Yes"},
                                        {"label": "No", "value": "No"}
                                    ], placeholder="Select option", className="form-control")
                                ]),
                                dbc.Col([
                                    html.Label("Has Gold Loan?", className="form-label"),
                                    dcc.Dropdown(id="input-gl-flag", options=[
                                        {"label": "Yes", "value": "Yes"},
                                        {"label": "No", "value": "No"}
                                    ], placeholder="Select option", className="form-control")
                                ]),
                            ])
                        ], className="form-step"),

                        html.Div([
                            html.H5([
                                html.I(className="fas fa-history me-2"),
                                "Step 4: Credit Info"
                            ]),
                            html.Hr(),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Credit Score", className="form-label"),
                                    dcc.Input(id="input-credit-score", type="number", min=300, max=850, placeholder="e.g. 720", className="form-control")
                                ]),
                                dbc.Col([
                                    html.Label("First Product Enquired", className="form-label"),
                                    dcc.Dropdown(id="input-first-prod-enq", options=[
                                        {"label": p, "value": p} for p in ["Credit Card", "Personal Loan", "Home Loan", "Gold Loan", "others"]
                                    ], placeholder="Select product", className="form-control")
                                ]),
                                dbc.Col([
                                    html.Label("Last Product Enquired", className="form-label"),
                                    dcc.Dropdown(id="input-last-prod-enq", options=[
                                        {"label": p, "value": p} for p in ["Credit Card", "Personal Loan", "Home Loan", "Gold Loan", "others"]
                                    ], placeholder="Select product", className="form-control")
                                ]),
                            ])
                        ], className="form-step"),

                        html.Div([
                            dbc.Row([
                                dbc.Col([
                                    dcc.Dropdown(
                                        id="model-selection",
                                        options=[
                                            {"label": "Random Forest", "value": "random_forest"},
                                            {"label": "XGBoost", "value": "xgboost"}
                                        ],
                                        value="random_forest",
                                        placeholder="Select model",
                                        className="model-selector",
                                        style={"margin-bottom": "1rem"}
                                    ),
                                ], width={"size": 6, "offset": 3})
                            ], className="justify-content-center mb-3"),
                            
                            html.Div([
                                dbc.Button("Predict Approval", id="predict-button", color="primary", className="predict-button")
                            ], className="text-center")
                        ], className="model-selection-container")
                    ])
                ])
            ], width=12, lg=6),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Prediction Result", className="card-title mb-4"),
                        dcc.Loading(
                            id="loading-prediction",
                            type="circle",
                            children=html.Div([
                                html.Div(id="prediction-output"),
                                html.Div(id="prediction-explanation", className="mt-4")
                            ])
                        )
                    ])
                ], className="prediction-card")
            ], width=12, lg=6)
        ], className="g-4 form-section")
    ], id="predict", **{"aria-label": "Prediction Form Section"})

def render_compare_page():
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Credit Best Practices & Tips", className="card-title mb-4"),
                    
                    dbc.ButtonGroup([
                        dbc.Button("Payment History", id="tip-1-btn", className="tip-btn", n_clicks=0),
                        dbc.Button("Credit Utilization", id="tip-2-btn", className="tip-btn", n_clicks=0),
                        dbc.Button("Credit Mix", id="tip-3-btn", className="tip-btn", n_clicks=0),
                        dbc.Button("New Credit", id="tip-4-btn", className="tip-btn", n_clicks=0),
                        dbc.Button("Monitoring", id="tip-5-btn", className="tip-btn", n_clicks=0),
                    ], className="mb-4 d-flex flex-wrap gap-2"),
                    
                    html.Div(id="dynamic-tips-content", className="tips-content"),
                    
                ])
            ], className="comparison-card")
        ], width=12, lg=10)
    ], className="g-4 justify-content-center")

def render_about_page():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("About This Project", className="card-title mb-4"),
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Team Members", className="text-primary mb-3"),
                                html.Ul([
                                    html.Li([html.I(className="fas fa-user me-2 text-success"), "Marius Francois Grassman"]),
                                    html.Li([html.I(className="fas fa-user me-2 text-success"), "Dieter Olivier"]),
                                    html.Li([html.I(className="fas fa-user me-2 text-success"), "Tiaan Dortfling"]),
                                    html.Li([html.I(className="fas fa-user me-2 text-success"), "Ryan Andrews"]),
                                    html.Li([html.I(className="fas fa-user me-2 text-success"), "Reghard du Plessis"]),
                                ], className="tips-list mb-4")
                            ])
                        ], className="tips-card fade-in mb-4"),

                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Project Overview", className="text-primary mb-3"),
                                html.P([html.I(className="fas fa-project-diagram me-2 text-success"), 
                                      "Credit Risk Assessment: Build a model to assess the creditworthiness of individuals or businesses and predict the risk of default on loans or credit lines."], 
                                      className="tips-list")
                            ])
                        ], className="tips-card fade-in mb-4"),

                        dbc.Card([
                            dbc.CardBody([
                                html.H5("How It Works", className="text-primary mb-3"),
                                html.P([html.I(className="fas fa-cogs me-2 text-success"), 
                                      "We developed a machine learning solution that predicts the likelihood of loan approval using user-provided financial and personal data. The app uses two powerful algorithms — Random Forest and XGBoost — to analyze risk based on patterns found in training data."],
                                      className="tips-list")
                            ])
                        ], className="tips-card fade-in mb-4"),

                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Our Models", className="text-primary mb-3"),
                                html.Ul([
                                    html.Li([html.I(className="fas fa-tree me-2 text-success"), 
                                           html.B("Random Forest: "), 
                                           "An ensemble of decision trees that reduces overfitting by averaging the results of many smaller trees. It's known for stability and handling mixed feature types."]),
                                    html.Li([html.I(className="fas fa-bolt me-2 text-success"), 
                                           html.B("XGBoost: "), 
                                           "A high-performance gradient boosting technique that builds trees sequentially, optimizing each one to correct errors made by previous trees. It's fast, accurate, and often used in real-world financial applications."]),
                                ], className="tips-list mb-4")
                            ])
                        ], className="tips-card fade-in mb-4"),

                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Why It Matters", className="text-primary mb-3"),
                                html.P([html.I(className="fas fa-chart-line me-2 text-success"), 
                                      "Credit risk assessment is critical for financial institutions. By automating this process with machine learning, we can provide faster, more consistent decisions — reducing human error and helping lenders manage risk more effectively."],
                                      className="tips-list")
                            ])
                        ], className="tips-card fade-in")
                    ])
                ], className="about-card")
            ], width=12, lg=8, className="mx-auto")
        ])
    ], fluid=False, className="py-4")

@app.callback(
    Output('model-selection', 'style'),
    Input('show-model-selection', 'value')
)
def toggle_model_selector(show):
    if show == 1:
        return {'display': 'block'}
    return {'display': 'none'}

@app.callback(
    Output("dynamic-tips-content", "children"),
    [Input(f"tip-{i}-btn", "n_clicks") for i in range(1,6)],
    prevent_initial_call=True
)
def update_tips_content(*args):
    ctx = dash.callback_context
    if not ctx.triggered:
        return ""

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    tip_content = {
        "tip-1-btn": dbc.Card([
            dbc.CardBody([
                html.H5("Payment History Best Practices", className="text-primary mb-3"),
                html.Ul([
                    html.Li([html.I(className="fas fa-check-circle me-2 text-success"), 
                           "Set up debit orders for all your accounts - South African banks like FNB, Standard Bank, and ABSA prefer automatic payments."]),
                    html.Li([html.I(className="fas fa-check-circle me-2 text-success"), 
                           "Always pay more than the minimum required amount on your credit card - aim for at least 15% above minimum."]),
                    html.Li([html.I(className="fas fa-check-circle me-2 text-success"), 
                           "Keep proof of payments for at least 5 years as per South African credit regulations."]),
                    html.Li([html.I(className="fas fa-check-circle me-2 text-success"), 
                           "Register for payment notifications with your bank to avoid missing due dates."]),
                    html.Li([html.I(className="fas fa-check-circle me-2 text-success"), 
                           "If you miss a payment, contact your creditor within 24 hours to make arrangements - South African credit providers are often willing to negotiate."]),
                    html.Li([html.I(className="fas fa-check-circle me-2 text-success"), 
                           "Maintain a good standing with your cell phone contract payments as these are reported to credit bureaus in South Africa."])
                ], className="tips-list")
            ])
        ], className="tips-card fade-in"),
        
        "tip-2-btn": dbc.Card([
            dbc.CardBody([
                html.H5("Credit Utilization in South Africa", className="text-primary mb-3"),
                html.Ul([
                    html.Li([html.I(className="fas fa-check-circle me-2 text-success"), 
                           "Keep your credit utilization below 30% - South African credit bureaus heavily weight this factor."]),
                    html.Li([html.I(className="fas fa-check-circle me-2 text-success"), 
                           "Avoid using credit cards for cash withdrawals as South African banks charge high fees and interest."]),
                    html.Li([html.I(className="fas fa-check-circle me-2 text-success"), 
                           "Don't max out store cards from retailers like Woolworths, Truworths, or Mr Price."]),
                    html.Li([html.I(className="fas fa-check-circle me-2 text-success"), 
                           "Maintain a healthy balance between credit card and store account usage."]),
                    html.Li([html.I(className="fas fa-check-circle me-2 text-success"), 
                           "Consider consolidating multiple store cards into a single lower-interest credit card."]),
                    html.Li([html.I(className="fas fa-check-circle me-2 text-success"), 
                           "Review your credit limit increases carefully - banks often offer automatic increases."])
                ], className="tips-list")
            ])
        ], className="tips-card fade-in"),

        "tip-3-btn": dbc.Card([
            dbc.CardBody([
                html.H5("Credit Mix for South African Consumers", className="text-primary mb-3"),
                html.Ul([
                    html.Li([html.I(className="fas fa-check-circle me-2 text-success"), 
                           "Balance different types of credit: home loan (bond), vehicle finance, credit card, and store accounts."]),
                    html.Li([html.I(className="fas fa-check-circle me-2 text-success"), 
                           "Avoid multiple personal loans from different providers like African Bank, Capitec, or Direct Axis."]),
                    html.Li([html.I(className="fas fa-check-circle me-2 text-success"), 
                           "Consider a secured credit card from major banks as a safer credit-building option."]),
                    html.Li([html.I(className="fas fa-check-circle me-2 text-success"), 
                           "Don't rely heavily on unsecured loans which carry higher interest rates in South Africa."]),
                    html.Li([html.I(className="fas fa-check-circle me-2 text-success"), 
                           "Keep your oldest credit account active to maintain credit history length."]),
                    html.Li([html.I(className="fas fa-check-circle me-2 text-success"), 
                           "Avoid payday loans and mashonisas (loan sharks) at all costs."])
                ], className="tips-list")
            ])
        ], className="tips-card fade-in"),

        "tip-4-btn": dbc.Card([
            dbc.CardBody([
                html.H5("Managing New Credit in South Africa", className="text-primary mb-3"),
                html.Ul([
                    html.Li([html.I(className="fas fa-check-circle me-2 text-success"), 
                           "Space out credit applications by at least 6 months - multiple applications impact your credit score significantly in SA."]),
                    html.Li([html.I(className="fas fa-check-circle me-2 text-success"), 
                           "Complete affordability assessments honestly as required by the National Credit Act."]),
                    html.Li([html.I(className="fas fa-check-circle me-2 text-success"), 
                           "Ensure you have all required documents (proof of income, bank statements, ID) before applying."]),
                    html.Li([html.I(className="fas fa-check-circle me-2 text-success"), 
                           "Research interest rates across different banks - they can vary significantly in South Africa."]),
                    html.Li([html.I(className="fas fa-check-circle me-2 text-success"), 
                           "Be cautious of credit providers who don't do proper affordability checks."]),
                    html.Li([html.I(className="fas fa-check-circle me-2 text-success"), 
                           "Check if you qualify for special rates (like professional banking) before applying."])
                ], className="tips-list")
            ])
        ], className="tips-card fade-in"),

        "tip-5-btn": dbc.Card([
            dbc.CardBody([
                html.H5("Credit Monitoring in South Africa", className="text-primary mb-3"),
                html.Ul([
                    html.Li([html.I(className="fas fa-check-circle me-2 text-success"), 
                           "Get your free annual credit report from TransUnion, Experian, or Compuscan."]),
                    html.Li([html.I(className="fas fa-check-circle me-2 text-success"), 
                           "Register with the Credit Ombud if you spot any irregularities in your credit report."]),
                    html.Li([html.I(className="fas fa-check-circle me-2 text-success"), 
                           "Check for any fraudulent RICA or FICA registrations under your name."]),
                    html.Li([html.I(className="fas fa-check-circle me-2 text-success"), 
                           "Monitor your credit score monthly through your bank's app or credit bureau services."]),
                    html.Li([html.I(className="fas fa-check-circle me-2 text-success"), 
                           "Keep records of all credit applications and responses as required by the National Credit Act."]),
                    html.Li([html.I(className="fas fa-check-circle me-2 text-success"), 
                           "Set up fraud alerts with the South African Fraud Prevention Service (SAFPS)."])
                ], className="tips-list")
            ])
        ], className="tips-card fade-in")
    }

    return tip_content.get(button_id, "")
server = app.server 
if __name__ == '__main__':
   app.run(debug=False, port=8050)

