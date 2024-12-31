<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>README</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        .container {
            padding: 20px;
            max-width: 900px;
            margin: auto;
        }
        h1 {
            text-align: center;
            color: #333;
        }
        p {
            text-align: justify;
            margin: 15px 0;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        td {
            border: 1px solid #ddd;
            padding: 10px;
            text-align: center;
        }
        img {
            max-width: 100%;
            height: auto;
        }
        video {
            max-width: 100%;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Data-Driven Insights for the Food and Beverage Industry</h1>
        <p>Data-driven decision-making is growing in importance in the food and beverage industry. With consumers increasingly relying on online reviews and ratings to choose restaurants, businesses can use this valuable data to optimize critical aspects like menu design, pricing strategies, services, and the overall customer experience. By providing actionable business intelligence, our system explores the relationship between restaurant ratings and factors such as cuisine type, pricing, and customer reviews.</p>
        <p>Through comprehensive Exploratory Data Analysis (EDA), significant patterns and trends influencing customer preferences and the restaurant industry can be uncovered. For rating prediction, we employ a variety of machine-learning models, including the <strong>ExtraTreesRegressor</strong>, which helps restaurant owners predict customer ratings and tailor their offerings to meet customer expectations better. Additionally, a recommendation system uses <strong>TF-IDF</strong> to suggest restaurants based on customer preferences for cuisine, price, reviews, and other key factors.</p>
        <p>This data-driven approach equips restaurant owners with powerful insights, enabling them to make informed decisions that enhance customer satisfaction and overall business performance while helping customers find the best dining options tailored to their tastes.</p>
        
        <table>
            <tr>
                <td><img src="o1.png" alt="Visualization 1"></td>
                <td><img src="o2.png" alt="Visualization 2"></td>
            </tr>
            <tr>
                <td>
                    <video controls>
                        <source src="rating_app.mp4" type="video/mp4">
                        Your browser does not support the video tag.
                    </video>
                </td>
                <td>
                    <video controls>
                        <source src="recom_app.mp4" type="video/mp4">
                        Your browser does not support the video tag.
                    </video>
                </td>
            </tr>
        </table>
    </div>
</body>
</html>
