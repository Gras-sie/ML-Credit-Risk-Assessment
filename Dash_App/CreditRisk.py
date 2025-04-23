import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import pickle
import os

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    external_scripts=[
        'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/js/all.min.js'
    ],
    suppress_callback_exceptions=True
)
app.title = "Credit Risk Predictor"

# === Paths ===
current_dir = os.path.dirname(os.path.abspath(__file__))
risk_assessment_dir = os.path.dirname(current_dir)  # Get Risk_Assessment directory
model_paths = {
    'random_forest': os.path.join(risk_assessment_dir, 'Artifacts', 'PLK', 'random_forest_model.pkl'),
    'xgboost': os.path.join(risk_assessment_dir, 'Artifacts', 'PLK', 'xgboost_model.pkl')
}
data_file_path = os.path.join(risk_assessment_dir, 'SRC', 'credit_risk_features.csv')
columns_path = os.path.join(risk_assessment_dir, 'Artifacts', 'PLK', 'training_columns.pkl')

# Add error handling for file loading
try:
    # Load models
    models = {}
    for key, path in model_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        with open(path, 'rb') as f:
            models[key] = pickle.load(f)

    # Load training columns
    if not os.path.exists(columns_path):
        raise FileNotFoundError(f"Columns file not found: {columns_path}")
    with open(columns_path, 'rb') as f:
        training_columns = pickle.load(f)

    # Load data file
    if not os.path.exists(data_file_path):
        raise FileNotFoundError(f"Data file not found: {data_file_path}")
    full_data = pd.read_csv(data_file_path)

except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please ensure all required files are in the correct directories:")
    print(f"Models should be in: {os.path.dirname(model_paths['random_forest'])}")
    print(f"Data should be in: {os.path.dirname(data_file_path)}")
    raise

# === Helper ===
def flag_to_risk(flag):
    return {
        "P1": "Low Risk - Likely Approved",
        "P2": "Medium Risk - Review Needed",
        "P3": "High Risk - Likely Denied",
        "P4": "Critical Risk - Must Deny"
    }.get(flag, "Unknown")

# === Prediction Callback (Tab 1) ===
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

    if not n_clicks or None in [age, gender, marital, education, income, employment_time, 
                                cc_flag, pl_flag, hl_flag, gl_flag, credit_score]:
        return html.Div("Please fill in all fields", className="text-danger"), ""

    # Use default model if missing
    if model_type not in models:
        model_type = 'random_forest'
    model = models[model_type]

    # Pull a base row from dataset
    base_row = full_data.sample(n=1, random_state=42).copy()

    # Overwrite with inputs
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
        'first_prod_enq2': first_prod_enq or 'Unknown',  # Handle missing values
        'last_prod_enq2': last_prod_enq or 'Unknown'    # Handle missing values
    }
    for col, val in input_mappings.items():
        if col in base_row.columns:
            base_row[col] = val

    # Drop target column if it exists
    base_row = base_row.drop(columns=['Approved_Flag'], errors='ignore')

    # Encode and align with training data
    if not isinstance(training_columns, list):  # Ensure training_columns is a list
        raise ValueError("Training columns must be a list.")
    input_encoded = pd.get_dummies(base_row)
    input_encoded = input_encoded.reindex(columns=training_columns, fill_value=0)

    # Predict and map output
    prediction_raw = model.predict(input_encoded)[0]
    label_map = {0: "P1", 1: "P2", 2: "P3", 3: "P4"}
    prediction = label_map.get(prediction_raw, prediction_raw)
    risk = flag_to_risk(prediction)

    # Return styled result
    return html.H3(f"Risk Assessment: {risk}", className="text-info"), html.P(
        f"Based on your application, the {model_type.replace('_', ' ').title()} model classified you as: {risk}. "
        f"This classification considers multiple financial and credit-based indicators."
    )

# === Model Comparison Callback (Tab 2) ===
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

# === Layout ===
app.layout = dbc.Container([
    # Decorative elements using native Dash components
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
        }  # Fixed unmatched closing brace
    ),

    # Update nav items with proper attributes
    dbc.Navbar([
        html.Img(src="/assets/images/logo.png", height="30px", className="me-2"),
        dbc.NavbarBrand("Credit Risk Predictor", className="ms-2"),
        dbc.Nav(
            [
                dbc.NavLink("Predict", id="nav-predict", href="#predict", 
                           active=True, className="nav-link-current"),
                dbc.NavLink("Compare", id="nav-compare", href="#compare"),
                dbc.NavLink("About", id="nav-about", href="#about"),
            ],
            className="ms-auto",
            navbar=True,
        )
    ], color="primary", dark=True, sticky="top"),

    # Global loading animation
    dcc.Loading(
        id="global-loading",
        type="circle",
        fullscreen=True,
        children=html.Div(id="loading-placeholder")  # Added valid children
    ),
    
    html.Header([
        html.Div([
            html.H1("Credit Risk Predictor", className="display-4"),
            html.P("Estimate credit approval potential based on client details", className="lead"),
        ], className="header text-center py-4", **{"aria-label": "Application Header"})
    ]),

    # Main content with ARIA roles
    html.Main([
        html.Div(id="page-content", role="main", **{"aria-live": "polite"})
    ]),

    # Semantic footer
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
     Input("nav-about", "n_clicks")]
)
def toggle_active_nav(predict_clicks, compare_clicks, about_clicks):
    # Get the context of the triggered input
    ctx = dash.callback_context

    # Default to the "Predict" tab if no input is triggered
    if not ctx.triggered:
        return render_predict_page(), True, False, False

    # Determine which tab was clicked
    clicked_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if clicked_id == "nav-predict":
        return render_predict_page(), True, False, False
    elif clicked_id == "nav-compare":
        return render_compare_page(), False, True, False
    elif clicked_id == "nav-about":
        return render_about_page(), False, False, True

    # Fallback to the "Predict" tab
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
    # Get the context of the triggered input
    ctx = dash.callback_context

    # Default to "Predict" tab if no input is triggered
    if not ctx.triggered:
        return "nav-link-current", "nav-link", "nav-link"

    # Determine which tab was clicked
    clicked_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if clicked_id == "nav-predict":
        return "nav-link-current", "nav-link", "nav-link"
    elif clicked_id == "nav-compare":
        return "nav-link", "nav-link-current", "nav-link"
    elif clicked_id == "nav-about":
        return "nav-link", "nav-link", "nav-link-current"

    # Fallback to "Predict" tab
    return "nav-link-current", "nav-link", "nav-link"

def render_predict_page():
    return html.Section([
        dbc.Row([
            # Input Section (left)
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Applicant Information", className="card-title mb-4"),

                        # Step 1 - Personal
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

                        # Step 2 - Employment
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

                        # Step 3 - Product Flags
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

                        # Step 4 - Credit History
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

                        # Submit Button
                        html.Div([
                            dbc.Button("Predict Approval", id="predict-button", color="primary", className="predict-button")
                        ], className="predict-button-container")
                    ])
                ])
            ], width=12, lg=6),

            # Output Section (right)
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
                    html.H4("Credit Optimization Toolkit", className="card-title mb-4"),
                    
                    # Interactive Tips Buttons
                    dbc.ButtonGroup([
                        dbc.Button("Payment History", id="tip-1-btn", className="tip-btn", n_clicks=0),
                        dbc.Button("Credit Utilization", id="tip-2-btn", className="tip-btn", n_clicks=0),
                        dbc.Button("Credit Mix", id="tip-3-btn", className="tip-btn", n_clicks=0),
                        dbc.Button("New Credit", id="tip-4-btn", className="tip-btn", n_clicks=0),
                        dbc.Button("Monitoring", id="tip-5-btn", className="tip-btn", n_clicks=0),
                    ], className="mb-4 d-flex flex-wrap gap-2"),
                    
                    # Tips Content
                    html.Div(id="dynamic-tips-content", className="tips-content"),
                    
                    # Visual Progress Section
                    html.Div([
                        html.H5("Credit Health Meter", className="mt-4"),
                        dcc.Graph(id="credit-health-meter", config={'displayModeBar': False},  # Fixed initialization
                                  figure={  # Added default figure
                                      "data": [],
                                      "layout": {"title": "Credit Health Meter"}
                                  })
                    ], className="progress-section")
                ])
            ], className="comparison-card")
        ], width=12, lg=10)
    ], className="g-4 justify-content-center")

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
                    html.Li([html.I(className="fas fa-check-circle me-2 text-success"), "Always pay your accounts on time to avoid negative listings on your credit report."]),
                    html.Li([html.I(className="fas fa-check-circle me-2 text-success"), "Set up debit orders for consistent payments on loans and credit cards."]),
                    html.Li([html.I(className="fas fa-check-circle me-2 text-success"), "If you miss a payment, contact your creditor immediately to negotiate a repayment plan."]),
                    html.Li([html.I(className="fas fa-check-circle me-2 text-success"), "Avoid legal action by keeping your accounts up to date."]),
                    html.Li([html.I(className="fas fa-check-circle me-2 text-success"), "Regularly check your credit report for any missed payments or errors."])
                ], className="tips-list")
            ])
        ], className="tips-card fade-in"),
        "tip-2-btn": dbc.Card([
            dbc.CardBody([
                html.H5("Optimizing Credit Utilization", className="text-primary mb-3"),
                html.Ul([
                    html.Li([html.I(className="fas fa-check-circle me-2 text-success"), "Keep your credit card balances below 30% of your credit limit."]),
                    html.Li([html.I(className="fas fa-check-circle me-2 text-success"), "Pay off your credit card balance in full each month to avoid interest charges."]),
                    html.Li([html.I(className="fas fa-check-circle me-2 text-success"), "Avoid maxing out your credit cards, as it negatively impacts your credit score."]),
                    html.Li([html.I(className="fas fa-check-circle me-2 text-success"), "Request a credit limit increase only if you can manage it responsibly."]),
                    html.Li([html.I(className="fas fa-check-circle me-2 text-success"), "Distribute your spending across multiple credit accounts to reduce utilization on any single account."])
                ], className="tips-list")
            ])
        ], className="tips-card fade-in"),
        "tip-3-btn": dbc.Card([
            dbc.CardBody([
                html.H5("Improving Credit Mix", className="text-primary mb-3"),
                html.Ul([
                    html.Li([html.I(className="fas fa-check-circle me-2 text-success"), "Maintain a mix of credit types, such as credit cards, personal loans, and home loans."]),
                    html.Li([html.I(className="fas fa-check-circle me-2 text-success"), "Avoid relying solely on short-term loans like payday loans, as they can harm your credit profile."]),
                    html.Li([html.I(className="fas fa-check-circle me-2 text-success"), "If you don’t have a credit history, consider opening a secured credit card to build one."]),
                    html.Li([html.I(className="fas fa-check-circle me-2 text-success"), "Keep older accounts open to show a longer credit history."]),
                    html.Li([html.I(className="fas fa-check-circle me-2 text-success"), "Avoid opening too many new accounts in a short period, as it may signal financial instability."])
                ], className="tips-list")
            ])
        ], className="tips-card fade-in"),
        "tip-4-btn": dbc.Card([
            dbc.CardBody([
                html.H5("Managing New Credit", className="text-primary mb-3"),
                html.Ul([
                    html.Li([html.I(className="fas fa-check-circle me-2 text-success"), "Only apply for new credit when absolutely necessary to avoid unnecessary inquiries on your credit report."]),
                    html.Li([html.I(className="fas fa-check-circle me-2 text-success"), "Shop around for the best interest rates within a short period to minimize the impact on your credit score."]),
                    html.Li([html.I(className="fas fa-check-circle me-2 text-success"), "Avoid opening store accounts for discounts unless you plan to use them responsibly."]),
                    html.Li([html.I(className="fas fa-check-circle me-2 text-success"), "Understand the terms and conditions of any new credit agreement before signing."]),
                    html.Li([html.I(className="fas fa-check-circle me-2 text-success"), "Ensure you can afford the monthly repayments before taking on new credit."])
                ], className="tips-list")
            ])
        ], className="tips-card fade-in"),
        "tip-5-btn": dbc.Card([
            dbc.CardBody([
                html.H5("Credit Monitoring Strategies", className="text-primary mb-3"),
                html.Ul([
                    html.Li([html.I(className="fas fa-check-circle me-2 text-success"), "Check your credit report regularly through South African credit bureaus like TransUnion or Experian."]),
                    html.Li([html.I(className="fas fa-check-circle me-2 text-success"), "Dispute any inaccuracies on your credit report immediately to avoid negative impacts."]),
                    html.Li([html.I(className="fas fa-check-circle me-2 text-success"), "Set up fraud alerts if you suspect identity theft or unauthorized credit activity."]),
                    html.Li([html.I(className="fas fa-check-circle me-2 text-success"), "Use credit monitoring services to receive real-time updates on changes to your credit profile."]),
                    html.Li([html.I(className="fas fa-check-circle me-2 text-success"), "Keep your personal information secure to prevent identity theft."])
                ], className="tips-list")
            ])
        ], className="tips-card fade-in")
    }

    return tip_content.get(button_id, "")

def render_about_page():
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("About", className="card-title mb-4"),
                    html.P("This section will contain information about the Credit Risk Predictor app, its purpose, and the team behind it. (To be updated)")
                ])
            ], className="about-card")
        ], width=12, lg=8)
    ], className="g-4 justify-content-center")

# === Callback to Show/Hide Model Selection ===
@app.callback(
    Output('model-selection', 'style'),
    Input('show-model-selection', 'value')
)
def toggle_model_selector(show):
    if show == 1:
        return {'display': 'block'}
    return {'display': 'none'}

# === Run the app ===
server = app.server  # <-- Add this line for Render compatibility
if __name__ == '__main__':
    app.run(debug=False, port=8050)  # Set debug=False for production
