import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
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
    children=[
        html.H1(
            className='title bg-grey',
            children=[
                'AI Classification of Unseen News Headlines'
            ]
        ),
        html.Div(
            className='row',
            children=[
                html.Div(
                    className='pretty_container eight columns offset-by-two columns',
                    children=[dcc.Input(
                        id="input_{}".format("text"), type="text",
                        placeholder="What is your Headline?",
                ),
                    ]
                ),
            ],
        ),
        html.Div(className='row',
                 children=[
                     html.Div(
                         className='pretty_container eight columns offset-by-two columns',
                         children=[
                             html.Div(id="out-all-types")
                         ]
                     )
                 ])
    ]
)

@app.callback(
    Output("out-all-types", "children"),
    [Input("input_{}".format("text"), "value")],
)
def cb_render(input_string):
    badofwords_validation = vectorizer.transform([str(input_string)])
    X_validation = badofwords_validation.toarray()
    predictions = model.predict(X_validation)

    if predictions[0] == 'b':
        predictions = 'Economics & Politics'

    if predictions[0] == 't':
        predictions = 'Science & Technology'

    if predictions[0] == 'm':
        predictions = 'Health & Health Care'

    if predictions[0] == 'e':
        predictions = 'Entertainment'

    return predictions


if __name__ == "__main__":
    app.run_server(debug=False)
