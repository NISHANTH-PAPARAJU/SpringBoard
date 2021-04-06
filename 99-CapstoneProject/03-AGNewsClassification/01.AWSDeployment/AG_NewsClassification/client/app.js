// function getBathValue() {
//   var uiBathrooms = document.getElementsByName("uiBathrooms");
//   for (var i in uiBathrooms) {
//     if (uiBathrooms[i].checked) {
//       return parseInt(i) + 1;
//     }
//   }
//   return -1; // Invalid Value
// }

// function getBHKValue() {
//   var uiBHK = document.getElementsByName("uiBHK");
//   for (var i in uiBHK) {
//     if (uiBHK[i].checked) {
//       return parseInt(i) + 1;
//     }
//   }
//   return -1; // Invalid Value
// }

function onClickedPredictCategory() {
  console.log("Predict Category button clicked");
  var headline = document.getElementById("uiheadline");
  var short_desc = document.getElementById("uidescription");
  var estCategory = document.getElementById("uiEvaluatedResult");

  // var url = "http://127.0.0.1:5000/predict_home_price"; 
  var url = "/api/predict_article_category";

  $.post(url, {
    headline: headline.value,
    shortdesc: short_desc.value,
  }, function (data, status) {
    console.log(data.predicted_category);
    console.log(data.class_confidences);
    estCategory.innerHTML = "<h2>With a confidence of : <strong>" + data.confidence + "%</strong>, the article might belong to : <strong>" + data.predicted_category + "</strong> category." + " </h2>";
    console.log(status);
  });
}

// function onPageLoad() {
//   console.log("document loaded");
//   // var url = "http://127.0.0.1:5000/get_location_names"; 
//   var url = "/api/get_location_names"; 
//   $.get(url, function (data, status) {
//     console.log("got response for get_location_names request");
//     if (data) {
//       var locations = data.locations;
//       var uiLocations = document.getElementById("uiLocations");
//       $('#uiLocations').empty();
//       for (var i in locations) {
//         var opt = new Option(locations[i]);
//         $('#uiLocations').append(opt);
//       }
//     }
//   });
// }

// window.onload = onPageLoad;
