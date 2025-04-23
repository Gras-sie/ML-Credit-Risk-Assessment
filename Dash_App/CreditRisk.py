import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import pickle
import os

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Credit Risk Predictor"


# === Paths ===
current_dir = os.path.dirname(os.path.abspath(__file__))
model_paths = {
    'random_forest': os.path.join(current_dir, '..', 'Artifacts', 'PLK', 'random_forest_model.pkl'),
    'xgboost': os.path.join(current_dir, '..', 'Artifacts', 'PLK', 'xgboost_model.pkl')
}
data_file_path = os.path.join(current_dir, '..', 'SRC', 'credit_risk_features.csv')
columns_path = os.path.join(current_dir, '..', 'Artifacts', 'PLK', 'training_columns.pkl')

# === Load models and training columns ===
models = {}
for key, path in model_paths.items():
    with open(path, 'rb') as f:
        models[key] = pickle.load(f)

with open(columns_path, 'rb') as f:
    training_columns = pickle.load(f)

full_data = pd.read_csv(data_file_path)

# === Helper ===
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
        'first_prod_enq2': first_prod_enq or 'Personal Loan',
        'last_prod_enq2': last_prod_enq or 'Credit Card'
    }
    for col, val in input_mappings.items():
        if col in base_row.columns:
            base_row[col] = val

    # Drop target column if it exists
    base_row = base_row.drop(columns=['Approved_Flag'], errors='ignore')

    # Encode and align with training data
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

app.layout = dbc.Container([

    # Page Header
    dbc.Row(dbc.Col(
        html.Div([
            html.H1("Credit Risk Predictor", className="display-4 mb-3"),
            html.H4("Estimate credit approval potential based on client details", className="text-light")
        ], className="header text-center")
    )),

    dbc.Row([
        # Input Section (left)
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Applicant Information", className="card-title mb-4"),

                    # Step 1 - Personal
                    html.H5("Step 1: Personal Details", className="mt-3"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Age"),
                            dcc.Input(id="input-age", type="number", min=18, max=100, placeholder="Enter age", className="form-control")
                        ], width=4),
                        dbc.Col([
                            html.Label("Gender"),
                            dcc.Dropdown(id="input-gender", options=[
                                {"label": "Male", "value": "Male"},
                                {"label": "Female", "value": "Female"},
                                {"label": "Other", "value": "Other"}
                            ], placeholder="Select gender", className="form-control")
                        ], width=4),
                        dbc.Col([
                            html.Label("Marital Status"),
                            dcc.Dropdown(id="input-marital", options=[
                                {"label": m, "value": m} for m in ["Single", "Married", "Divorced", "Widowed"]
                            ], placeholder="Marital Status", className="form-control")
                        ], width=4),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Education"),
                            dcc.Dropdown(id="input-education", options=[
                                {"label": edu, "value": edu} for edu in ["High School", "Diploma", "Bachelor's", "Master's", "PhD", "Other"]
                            ], placeholder="Education Level", className="form-control")
                        ], width=12)
                    ], className="mb-3"),

                    # Step 2 - Employment
                    html.H5("Step 2: Employment & Income", className="mt-3"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Net Monthly Income (ZAR)"),
                            dcc.Input(id="input-income", type="number", placeholder="e.g. 10000", className="form-control")
                        ], width=6),
                        dbc.Col([
                            html.Label("Time With Current Employer (Months)"),
                            dcc.Input(id="input-employment-time", type="number", placeholder="e.g. 24", className="form-control")
                        ], width=6)
                    ], className="mb-3"),

                    # Step 3 - Product Flags
                    html.H5("Step 3: Loan Ownership", className="mt-3"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Has Credit Card?"),
                            dcc.Dropdown(id="input-cc-flag", options=[
                                {"label": "Yes", "value": "Yes"},
                                {"label": "No", "value": "No"}
                            ], placeholder="Select option", className="form-control")
                        ], width=3),
                        dbc.Col([
                            html.Label("Has Personal Loan?"),
                            dcc.Dropdown(id="input-pl-flag", options=[
                                {"label": "Yes", "value": "Yes"},
                                {"label": "No", "value": "No"}
                            ], placeholder="Select option", className="form-control")
                        ], width=3),
                        dbc.Col([
                            html.Label("Has Home Loan?"),
                            dcc.Dropdown(id="input-hl-flag", options=[
                                {"label": "Yes", "value": "Yes"},
                                {"label": "No", "value": "No"}
                            ], placeholder="Select option", className="form-control")
                        ], width=3),
                        dbc.Col([
                            html.Label("Has Gold Loan?"),
                            dcc.Dropdown(id="input-gl-flag", options=[
                                {"label": "Yes", "value": "Yes"},
                                {"label": "No", "value": "No"}
                            ], placeholder="Select option", className="form-control")
                        ], width=3),
                    ], className="mb-3"),

                    # Step 4 - Credit History
                    html.H5("Step 4: Credit Info", className="mt-3"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Credit Score"),
                            dcc.Input(id="input-credit-score", type="number", min=300, max=850, placeholder="e.g. 720", className="form-control")
                        ], width=4),
                        dbc.Col([
                            html.Label("First Product Enquired"),
                            dcc.Dropdown(id="input-first-prod-enq", options=[
                                {"label": p, "value": p} for p in ["Credit Card", "Personal Loan", "Home Loan", "Gold Loan", "others"]
                            ], placeholder="Select product", className="form-control")
                        ], width=4),
                        dbc.Col([
                            html.Label("Last Product Enquired"),
                            dcc.Dropdown(id="input-last-prod-enq", options=[
                                {"label": p, "value": p} for p in ["Credit Card", "Personal Loan", "Home Loan", "Gold Loan", "others"]
                            ], placeholder="Select product", className="form-control")
                        ], width=4),
                    ], className="mb-3"),

                    # Model Toggle & Submit
                    dbc.Row([
                        dbc.Col([
                            html.Label("Show Model Selection"),
                            dcc.Dropdown(
                                id='show-model-selection',
                                options=[{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}],
                                value=0,
                                className="form-control"
                            )
                        ], width=12),
                        dbc.Col([
                            html.Label("Model Selection"),
                            dcc.Dropdown(
                                id='model-selection',
                                options=[
                                    {'label': 'Random Forest', 'value': 'random_forest'},
                                    {'label': 'XGBoost Classifier', 'value': 'xgboost'}
                                ],
                                value='random_forest',
                                className="form-control",
                                style={'display': 'none'}
                            )
                        ], width=12),
                        dbc.Col([
                            dbc.Button("Predict Approval", id="predict-button", color="primary", className="mt-4 w-100")
                        ])
                    ])
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
    ])
], fluid=True, className="app-container")

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
