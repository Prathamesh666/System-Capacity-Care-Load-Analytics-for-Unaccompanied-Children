from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LinearRegression, HuberRegressor, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier
from scipy.interpolate import griddata
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import xgboost as xgb
import warnings

def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    """Train regression models safely with NaN handling, return figures + performance metrics."""

    # --- Handle NaNs globally ---
    imputer = SimpleImputer(strategy="mean")   # replace NaNs with column mean
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "XGBoost Regressor": xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "SVR": SVR(kernel='rbf', C=100, gamma=0.1),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
        "K-Neighbors Regressor": KNeighborsRegressor(n_neighbors=5),
        "Huber Regressor": HuberRegressor(max_iter=5000, alpha=0.0001)
    }

    performance_data = {"MAE": {}, "RMSE": {}}
    figures = {}

    # Scale for Huber
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for name, model in models.items():
        if name == "Huber Regressor":
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        performance_data["MAE"][name] = mae
        performance_data["RMSE"][name] = rmse

        # Plotly chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test.index, y=y_test, mode='lines', name='Actual', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=y_test.index, y=y_pred, mode='lines', name=f'{name} Predictions', line=dict(dash='dash')))
        fig.update_layout(
            title=dict(text=f'{name}: Actual vs Predicted', x=0.5, xanchor='center'),
            xaxis_title='Date', yaxis_title='Total System Load',
            legend=dict(x=0.01, y=0.99, bordercolor="Black", borderwidth=1)
        )
        figures[name] = fig

    # Performance DataFrame
    performance_df = pd.DataFrame(performance_data)

    # Heatmap
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=performance_df.values,
        x=performance_df.columns,
        y=performance_df.index,
        colorscale='Viridis_r',
        text=performance_df.values,
        texttemplate="%{text:.2f}",
        hovertemplate="Model: %{y}<br>Metric: %{x}<br>Value: %{z:.2f}<extra></extra>"
    ))
    fig_heatmap.update_layout(
        title=dict(text='<b>Regression Model Performance Comparison</b>', x=0.5, xanchor='center'),
        xaxis_title='Metric', yaxis_title='Model', autosize=True
    )

    figures["Performance Heatmap"] = fig_heatmap

    return figures, performance_df

# -------------------------
# Common function
# -------------------------
def train_and_evaluate_classifiers(X_train_clf_scaled, y_train_clf, X_test_clf_scaled, y_test_clf, label_encoder):
    classifiers = {
        'Logistic Regression': OneVsRestClassifier(LogisticRegression(random_state=42, solver='liblinear')),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(eval_metric='mlogloss', random_state=42),
        'SVC': SVC(probability=True, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'K-Neighbors': KNeighborsClassifier(),
        'Gaussian Naive Bayes': GaussianNB()
    }

    results = []
    all_encoded_labels = label_encoder.transform(label_encoder.classes_)

    for name, clf in classifiers.items():
        clf.fit(X_train_clf_scaled, y_train_clf)
        y_pred = clf.predict(X_test_clf_scaled)

        accuracy = accuracy_score(y_test_clf, y_pred)
        precision = precision_score(y_test_clf, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test_clf, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test_clf, y_pred, average='weighted', zero_division=0)

        # --- ROC AUC ---
        roc_auc = np.nan
        if hasattr(clf, "predict_proba"):
            y_proba = clf.predict_proba(X_test_clf_scaled)
            unique_classes_in_y_test = np.unique(y_test_clf)

            if len(unique_classes_in_y_test) == 2:
                # Binary classification → use positive class column
                roc_auc = roc_auc_score(y_test_clf, y_proba[:, 1])
            elif len(unique_classes_in_y_test) > 2:
                # Multiclass classification → use full probability matrix
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Only one class is present in y_true.", category=UserWarning)
                    roc_auc = roc_auc_score(y_test_clf, y_proba, multi_class='ovr', average='weighted', labels=all_encoded_labels)

        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC AUC': roc_auc
        })

    classification_performance_df = pd.DataFrame(results)

    # --- Heatmap ---
    performance_df_clf_indexed = classification_performance_df.set_index('Model')
    fig_heatmap_clf = go.Figure(data=go.Heatmap(
        z=performance_df_clf_indexed.values,
        x=performance_df_clf_indexed.columns,
        y=performance_df_clf_indexed.index,
        colorscale='Viridis',
        text=performance_df_clf_indexed.values,
        texttemplate="%{text:.3f}",
        hovertemplate="Model: %{y}<br>Metric: %{x}<br>Value: %{z:.3f}<extra></extra>"
    ))
    fig_heatmap_clf.update_layout(
        title=dict(text='<b>Classification Model Performance Comparison</b>', x=0.5, xanchor='center', font=dict(size=17)),
        xaxis_title='Metric', yaxis_title='Model',
        template='plotly_white', autosize=True
    )

    # --- 3D Scatter ---
    df_melted = classification_performance_df.melt(id_vars=['Model'], var_name='Metric', value_name='Score')
    model_names = df_melted['Model'].unique()
    metric_names = df_melted['Metric'].unique()
    model_mapping = {model: i for i, model in enumerate(model_names)}
    metric_mapping = {metric: i for i, metric in enumerate(metric_names)}
    df_melted['Model_Numeric'] = df_melted['Model'].map(model_mapping)
    df_melted['Metric_Numeric'] = df_melted['Metric'].map(metric_mapping)
    df_plot_3d = df_melted.dropna(subset=['Score'])

    fig_3d_scatter = go.Figure(data=[go.Scatter3d(
        x=df_plot_3d['Model_Numeric'],
        y=df_plot_3d['Metric_Numeric'],
        z=df_plot_3d['Score'],
        mode='markers',
        marker=dict(size=10, color=df_plot_3d['Score'], colorscale='Viridis', opacity=0.8,
                    colorbar=dict(title='<b>Performance Score</b>', x=1.05)),
        text=[f'Model: {m}<br>Metric: {mt}<br>Score: {s:.3f}' for m, mt, s in zip(df_plot_3d['Model'], df_plot_3d['Metric'], df_plot_3d['Score'])],
        hoverinfo='text'
    )])
    fig_3d_scatter.update_layout(
        title=dict(text='<b>3D Scatter Plot: Classification Model Performance</b>', x=0.5, xanchor='center', font=dict(size=17)),
        scene=dict(
            xaxis_title='Model', yaxis_title='Metric', zaxis_title='Performance Score',
            xaxis=dict(tickmode='array', tickvals=list(model_mapping.values()), ticktext=list(model_mapping.keys())),
            yaxis=dict(tickmode='array', tickvals=list(metric_mapping.values()), ticktext=list(metric_mapping.keys()))
        ), autosize=True
    )

    # --- 3D Surface ---
    x_coords = df_plot_3d['Model_Numeric'].values
    y_coords = df_plot_3d['Metric_Numeric'].values
    z_values = df_plot_3d['Score'].values
    grid_x, grid_y = np.mgrid[x_coords.min():x_coords.max():50j, y_coords.min():y_coords.max():50j]
    grid_z = griddata((x_coords, y_coords), z_values, (grid_x, grid_y), method='cubic')

    fig_3d_surface = go.Figure(data=[go.Surface(
        z=grid_z, x=grid_x, y=grid_y,
        colorscale='Viridis',
        colorbar=dict(title='<b>Performance Score</b>', x=1.05),
        cmin=z_values.min(), cmax=z_values.max()
    )])
    fig_3d_surface.update_layout(
        title=dict(text='<b>3D Heatmap Surface Plot: Classification Model Performance</b>', x=0.5, xanchor='center', font=dict(size=17)),
        scene=dict(
            xaxis_title='Model', yaxis_title='Metric', zaxis_title='Performance Score',
            xaxis=dict(tickmode='array', tickvals=list(model_mapping.values()), ticktext=list(model_mapping.keys())),
            yaxis=dict(tickmode='array', tickvals=list(metric_mapping.values()), ticktext=list(metric_mapping.keys()))
        ), autosize=True
    )

    figures = {
        "Performance Heatmap": fig_heatmap_clf,
        "3D Scatter": fig_3d_scatter,
        "3D Surface": fig_3d_surface
    }

    return figures, classification_performance_df

def iterative_forecast(model, X_historical, initial_target_series, future_X_template, scaler, num_steps):
    forecast_values = []
    current_target = initial_target_series.copy()

    all_feature_columns = X_historical.columns.tolist()

    # Features to be dynamically derived
    derived_features_list = [
        'Day_of_Week', 'Day_of_Month', 'Month', 'Year', 'Week_of_Year', 'Is_Weekend',
        'Day_of_Week_sin', 'Day_of_Week_cos', 'Month_sin', 'Month_cos',
        'Day_of_Month_sin', 'Day_of_Month_cos'
    ]
    derived_features_list.extend([f'Load_Lag_{i}d' for i in range(1, 8)])
    derived_features_list.extend([f'Load_Rolling_Mean_{w}d' for w in [7, 14, 30]])
    derived_features_list.extend([f'Load_Rolling_Std_{w}d' for w in [7, 14, 30]])

    # Handle duplicated rolling features
    if '7-Day Rolling Avg Load' in all_feature_columns:
        derived_features_list.append('7-Day Rolling Avg Load')
    if '14-Day Rolling Avg Load' in all_feature_columns:
        derived_features_list.append('14-Day Rolling Avg Load')
    if '7-Day Rolling Std Dev Load' in all_feature_columns:
        derived_features_list.append('7-Day Rolling Std Dev Load')

    features_to_set_to_zero = [c for c in all_feature_columns if c not in derived_features_list]

    for i in range(num_steps):
        pred_date = future_X_template.index[i]
        features_for_pred = pd.DataFrame(index=[pred_date], columns=all_feature_columns)

        # Populate date-based features
        for col in ['Day_of_Week','Day_of_Month','Month','Year','Week_of_Year','Is_Weekend',
                    'Day_of_Week_sin','Day_of_Week_cos','Month_sin','Month_cos',
                    'Day_of_Month_sin','Day_of_Month_cos']:
            if col in features_for_pred.columns:
                features_for_pred.loc[pred_date, col] = future_X_template.loc[pred_date, col]

        # Lag features
        for lag in range(1, 8):
            if f'Load_Lag_{lag}d' in features_for_pred.columns:
                features_for_pred.loc[pred_date, f'Load_Lag_{lag}d'] = (
                    current_target.iloc[-lag] if len(current_target) >= lag else np.nan
                )

        # Rolling features
        temp_series = pd.concat([current_target, pd.Series([np.nan], index=[pred_date])]).sort_index()
        for window in [7, 14, 30]:
            rolling_mean = temp_series.rolling(window).mean().shift(1).loc[pred_date]
            rolling_std = temp_series.rolling(window).std().shift(1).loc[pred_date]

            if f'Load_Rolling_Mean_{window}d' in features_for_pred.columns:
                features_for_pred.loc[pred_date, f'Load_Rolling_Mean_{window}d'] = rolling_mean
            if f'Load_Rolling_Std_{window}d' in features_for_pred.columns:
                features_for_pred.loc[pred_date, f'Load_Rolling_Std_{window}d'] = rolling_std

            if window == 7:
                if '7-Day Rolling Avg Load' in features_for_pred.columns:
                    features_for_pred.loc[pred_date, '7-Day Rolling Avg Load'] = rolling_mean
                if '7-Day Rolling Std Dev Load' in features_for_pred.columns:
                    features_for_pred.loc[pred_date, '7-Day Rolling Std Dev Load'] = rolling_std
            elif window == 14:
                if '14-Day Rolling Avg Load' in features_for_pred.columns:
                    features_for_pred.loc[pred_date, '14-Day Rolling Avg Load'] = rolling_mean

        # Set other features to zero
        for col in features_to_set_to_zero:
            if col in features_for_pred.columns:
                features_for_pred.loc[pred_date, col] = 0

        features_for_pred = features_for_pred[X_historical.columns]

        if features_for_pred.isnull().any().any():
            break

        scaled_features = scaler.transform(features_for_pred)
        predicted_value = model.predict(scaled_features)[0]
        forecast_values.append(predicted_value)
        current_target.loc[pred_date] = predicted_value

    return pd.Series(forecast_values, index=future_X_template.index[:len(forecast_values)])
