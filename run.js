// Define the URL of the server endpoint
const url = "http://localhost:8080/run-algorithm";

// Define the data to be sent in the request body
const inputData = [
  43, 53, 5, 5, 5, 5, 53, 5, 5, 5234, 5, 5, 5, 5, 5, 5, 512, 5, 5, 5, 5, 5, 5,
  5, 5, 5, 5, 5, 5, 55, 5, 5, 5, 5, 5, 5, 55,
];

// Send the POST request using fetch
fetch(url, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify(inputData),
})
  .then((response) => {
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
  })
  .then((data) => {
    console.log(`Probability of PCOS: ${data.probability_pcos * 100}%`);
  })
  .catch((error) => {
    console.error("Error:", error);
  });
