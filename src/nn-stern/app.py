#!/usr/bin/env python
from flask import Flask, jsonify, request
from pandas import DataFrame


app = Flask(__name__)

@app.route('/')
def index():
    return "Hello, captain John!"


@app.route('/price/1hour', methods=['POST'])
def json_example():
    req_data = request.get_json()

    doc = req_data['doc']
    name = req_data['name']
    namespace = req_data['namespace'] # two keys are needed because of the nested object
    pandas_version = req_data['pandas_version'] # an index is needed because of the array
    json_version = req_data['json_version']
    price = req_data['fields'][0]['tickvolume']

    original_df = DataFrame(req_data['fields'])
    print(original_df)

    return '''
        The doc value is: {}
        The name value is: {}
        The namespace version is: {}
        The pandas_version: {}
        The json_version: {}
        The Price is_ {}
        The number of records is: {:d}
        '''.format(
            doc, name, namespace, pandas_version, json_version,
            price, len(req_data['fields'])
            )


if __name__ == '__main__':
    app.run(debug=True, port=5000)
