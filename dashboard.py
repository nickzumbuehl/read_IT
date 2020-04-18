import dash
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objects as go
import joblib
import os
import pickle


model = pickle.load(open("classifier.p", "rb"))
vectorizer = pickle.load(open("vectorizer.p", "rb"))

input = ['microprocessors become cheaper']

badofwords_validation = vectorizer.transform(input)
X_validation = badofwords_validation.toarray()
predictions = model.predict(X_validation)


app = dash.Dash(__name__)
server = app.server
app.config.suppress_callback_exceptions = True


app.layout = html.Div(
    children=[html.H1(
        className='title bg-grey',
        children=['Read IT: AI Primer in Newspaper Classification']
    )]
)


if __name__ == "__main__":
    app.run_server(debug=False)