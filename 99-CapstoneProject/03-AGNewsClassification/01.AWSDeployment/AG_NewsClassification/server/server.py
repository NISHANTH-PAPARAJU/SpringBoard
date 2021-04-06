# from flask import Flask, request, jsonify
# import util

# app = Flask(__name__)

# @app.route('/get_location_names', methods=['GET'])
# def get_location_names():
#     response = jsonify({
#         'locations': util.get_location_names()
#     })
#     response.headers.add('Access-Control-Allow-Origin', '*')

#     return response

# @app.route('/predict_home_price', methods=['GET', 'POST'])
# def predict_home_price():
#     total_sqft = float(request.form['total_sqft'])
#     location = request.form['location']
#     bhk = int(request.form['bhk'])
#     bath = int(request.form['bath'])

#     response = jsonify({
#         'estimated_price': util.get_estimated_price(location,total_sqft,bhk,bath)
#     })
#     response.headers.add('Access-Control-Allow-Origin', '*')

#     return response

# if __name__ == "__main__":
#     print("Starting Python Flask Server For Home Price Prediction...")
#     util.load_saved_artifacts()
#     app.run()


from flask import Flask, request, jsonify
import util

app = Flask(__name__)


@app.route('/predict_article_category', methods=['GET', 'POST'])
def predict_article_category():
    headline = request.form['headline']
    short_desc = request.form['shortdesc']
    util.run_article_prediction(headline, short_desc)
    # response = jsonify({
    #     'predicted_category': util.get_predicted_cat(),
    #     'confidence': util.get_confidence_score(),
    #     'class_confidences': util.get_class_confidences()
    # })
    # response.headers.add('Access-Control-Allow-Origin', '*')

    response = jsonify({
        'predicted_category': util.get_predicted_cat(),
        'confidence': util.get_confidence_score(),
        'class_confidences': str(util.get_class_confidences())
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response


if __name__ == "__main__":
    print("Starting Python Flask Server for Article Category Prediction...")
    util.load_saved_artifacts()
    app.run()
