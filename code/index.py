# -*- coding: utf-8 -*-

import logging
import sklearn
import pickle
import json

def initializer(environ):
    pass

def handler(environ, start_response):
    with open("./model/creditcard_fraud_detection_model.pkl", "rb") as f:
        lr = pickle.load(f)

    context = environ['fc.context']
    request_uri = environ['fc.request_uri']
    request_method = environ['REQUEST_METHOD']
    try:
        request_body_size = int(environ.get('CONTENT_LENGTH', 0))
        request_body = environ['wsgi.input'].read(request_body_size)
    except (ValueError):
        request_body_size = 0
    if request_method == 'POST':
        #logger.info(request_body)
        print(request_body.decode("utf-8"))
        js = json.loads(request_body.decode("utf-8"))
        X = [js["Time"]]
        for i in range(1, 29):
            X.append(js["V{}".format(i)])
        X.append(js["Amount"])
        preds = lr.predict_proba([X])
        response = str(preds[0,1]).encode("utf-8")
    else:
        response = b'Hello World!!'
    # do something here
    status = '200 OK'
    response_headers = [('Content-type', 'text/plain')]
    start_response(status, response_headers)
    return [response]

