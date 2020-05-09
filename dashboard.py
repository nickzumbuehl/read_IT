import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import joblib
import os
import pickle
import requests
import json
import pandas as pd
import newsapi
from newsapi import NewsApiClient

# read models
model = pickle.load(open("classifier.p", "rb"))
vectorizer = pickle.load(open("vectorizer.p", "rb"))


app = dash.Dash(__name__)
server = app.server
app.config.suppress_callback_exceptions = True

# pulling data from newsapi:
newsapi = NewsApiClient(api_key='4ffaa0cb22f44814800f9b47f3fc176e')  # client key should be secret

# pulling some news here
everything_german = newsapi.get_everything(
    sources='handelsblatt,wirtschafts-woche,der-tagesspiegel,focus'
    # q='Flüchtling',
    # language='de',
)
df_in_german = pd.DataFrame(everything_german['articles'])

# logo.png

app.layout = html.Div(
    children=[
        html.Div(
            className='border_container twelve columns',
            children=[
                dcc.Tabs(
                    className='bare_container five columns offset-by-six column',
                    children=[
                        dcc.Tab(
                            className='custom-tab',
                            selected_className='custom-tab--selected',
                            label='YOUR NEWSPAPER',
                            children=[
                                html.Div(
                                    className='one-third column',
                                    children=[
                                        html.Img(
                                            className='pretty_container_two two columns',
                                            src='https://raw.githubusercontent.com/nickzumbuehl/read_IT/master/logo.png'),
                                        html.Div(
                                            className='pretty_container twelve columns',
                                                 children=[
                                                     html.Div(
                                                         className='row',
                                                         children=[html.Img(
                                                             className='pretty_container twelve columns',
                                                             src=df_in_german.urlToImage[0]
                                                         )]),
                                                     html.H5(
                                                         className='row',
                                                         children=[df_in_german.title[0]]),
                                                     html.H6(
                                                         className='row',
                                                         children=[df_in_german.description[0] + ' (Source: {})'.format(df_in_german.source[0]['name'])]),
                                                     html.Div(
                                                         className='row',
                                                         children=[
                                                             html.Div(
                                                                 className='bare_container four columns offset-by-eight columns',
                                                                 children=[html.A("ReadIT", href=df_in_german.url[0], target="_blank")]),
                                                         ]),
                                                 ]),
                                    ]),
                                html.Div(
                                    className='one-third column',
                                    children=[
                                        html.Div(
                                            className='pretty_container twelve columns',
                                                 children=[
                                                     html.Div(
                                                         className='row',
                                                         children=[html.Img(
                                                             className='pretty_container twelve columns',
                                                             src=df_in_german.urlToImage[1]
                                                         )]),
                                                     html.H5(
                                                         className='row',
                                                         children=[df_in_german.title[1]]),
                                                     html.H6(
                                                         className='row',
                                                         children=[df_in_german.description[1] + ' (Source: {})'.format(df_in_german.source[1]['name'])]),
                                                     html.Div(
                                                                      className='row',
                                                                      children=[
                                                                          html.Div(
                                                                              className='bare_container four columns offset-by-eight columns',
                                                                              children=[html.A("ReadIT", href=df_in_german.url[1], target="_blank")]),
                                                                      ]),
                                                              ]),
                                                 ]),
                                        html.Div(
                                                 className='one-third column',
                                                 children=[
                                                     html.Div(
                                                         className='pretty_container twelve columns',
                                                         children=[
                                                             html.Div(
                                                                 className='row',
                                                                 children=[html.Img(
                                                                     className='pretty_container twelve columns',
                                                                     src=df_in_german.urlToImage[3]
                                                                 )]),
                                                             html.H5(
                                                                 className='row',
                                                                 children=[df_in_german.title[3]]),
                                                             html.H6(
                                                                 className='row',
                                                                 children=[df_in_german.description[3] + ' (Source: {})'.format(df_in_german.source[3]['name'])]),
                                                             html.Div(
                                                                 className='row',
                                                                 children=[
                                                                     html.Div(
                                                                         className='bare_container four columns offset-by-eight columns',
                                                                         children=[html.A("ReadIT", href=df_in_german.url[3], target="_blank")]),
                                                                 ]),
                                                         ]),
                                                 ]),
                            ]),
                        dcc.Tab(
                            className='custom-tab',
                            selected_className='custom-tab--selected',
                            label='YOUR FILTER'),
                        dcc.Tab(className='custom-tab',
                                selected_className='custom-tab--selected',
                                label='VISIT SOURCE'),
                    ]),
                html.H6()
            ]),
        # html.H1(
        #     className='row bg-grey',
        #     children=[
        #         'AI Classification of Unseen News Headlines'
        #     ]
        # ),
        # html.Div(
        #     className='row',
        #     children=[
        #         html.Div(
        #             className='pretty_container ten columns offset-by-one columns',
        #             children=[
        #                 html.H2('The algorithm was trained on a data set consisting of Economics & Business,'
        #                         ' Science & Technology, Health & Health Care and Entertainment news articles from the'
        #                         ' United States. The AI aims to classify unseen news headlines according to these'
        #                         ' four clusters. Have fun testing it out!')
        #             ]
        #         )
        #     ]
        # ),
        # html.Div(
        #     className='row',
        #     children=[
        #         html.Div(
        #             className='pretty_container ten columns offset-by-one columns',
        #             children=[dcc.Input(
        #                 id="input_{}".format("text"), type="text",
        #                 placeholder="What is your Headline?",
        #             ),
        #             ]
        #         ),
        #     ],
        # ),
        # html.Div(className='row',
        #          children=[
        #              html.Div(
        #                  className='pretty_container ten columns offset-by-one columns',
        #                  children=[
        #                      html.Div(id="out-all-types")
        #                  ]
        #              )
        #          ]),
    ]
)

@app.callback(
    Output("out-all-types", "children"),
    [Input("input_{}".format("text"), "value")],
)
def cb_render(input_string):
    url_trans = 'https://api.deepl.com/v2/translate'
    payload = {'auth_key': 'e90be2dd-92b3-920e-5d78-be332af77f0b',
               'text': input_string,
               'target_lang': 'EN'}
    r = requests.get(url_trans, params=payload)
    output_text = json.loads(r.text)['translations'][0]
    input_string = output_text['text']

    badofwords_validation = vectorizer.transform([str(input_string)])
    X_validation = badofwords_validation.toarray()
    predictions = model.predict(X_validation)

    if predictions[0] == 'b':
        predictions = 'Economics & Business'

    if predictions[0] == 't':
        predictions = 'Science & Technology'

    if predictions[0] == 'm':
        predictions = 'Health & Health Care'

    if predictions[0] == 'e':
        predictions = 'Entertainment'

    return predictions


if __name__ == "__main__":
    app.run_server(debug=False)
