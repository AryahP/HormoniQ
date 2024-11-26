from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

hostName = "localhost"
serverPort = 8080

class MyServer(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        self.send_response(405)
        self.send_header("Content-type", "text/html")
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(bytes("<html><head><title>Method Not Allowed</title></head>", "utf-8"))
        self.wfile.write(bytes("<body><p>GET method is not supported. Please use POST.</p></body></html>", "utf-8"))


    def do_GET(self):
        self.send_response(405)
        self.send_header("Content-type", "text/html")
        self.end_headers()

    def do_POST(self):
        if self.path == "/run-algorithm":
            # Get the length of the data
            content_length = int(self.headers['Content-Length'])
            # Read the data
            post_data = self.rfile.read(content_length)
            # Parse the JSON data
            user_input = json.loads(post_data)

            # Validate the length of the input data
            expected_length = 37  # Replace with the actual number of features used during training
            if len(user_input) != expected_length:
                raise ValueError(f"Expected input length {expected_length}, but got {len(user_input)}")


            # Run the algorithm and get the result
            probability_pcos = self.run_algorithm(user_input)
            
            # Send the response
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            response = {
                "probability_pcos": probability_pcos
            }
            self.wfile.write(bytes(json.dumps(response), "utf-8"))
        else:
            self.send_response(404)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(bytes("<html><head><title>404 Not Found</title></head>", "utf-8"))
            self.wfile.write(bytes("<body><p>Page not found.</p></body></html>", "utf-8"))

    def run_algorithm(self, user_input):
        # reading the dataset
        file_path = r"C:\Users\jessi\OneDrive - Princeton University\Princeton Hacks\Diagnostic-Celiac-Disease-Management\PCOS Dataset.csv"
        try:
            data = pd.read_csv(file_path)
            print("Dataset loaded successfully.")
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return None

        # Cleaning data set (converting strings to numbers)
        data["AMH(ng/mL)"] = pd.to_numeric(data["AMH(ng/mL)"], errors='coerce')
        data["II    beta-HCG(mIU/mL)"] = pd.to_numeric(data["II    beta-HCG(mIU/mL)"], errors='coerce')

        data['Marraige Status (Yrs)'] = data['Marraige Status (Yrs)'].fillna(data['Marraige Status (Yrs)'].median())
        data['II    beta-HCG(mIU/mL)'] = data['II    beta-HCG(mIU/mL)'].fillna(data['II    beta-HCG(mIU/mL)'].median())
        data['AMH(ng/mL)'] = data['AMH(ng/mL)'].fillna(data['AMH(ng/mL)'].median())
        data['Fast food (Y/N)'] = data['Fast food (Y/N)'].fillna(data['Fast food (Y/N)'].mode()[0])

        # Clearing up the extra space in the column names (optional)
        data.columns = [col.strip() for col in data.columns]

        # Identifying non-numeric columns
        non_numeric_columns = data.select_dtypes(include=['object']).columns
        print("Non-numeric columns:", non_numeric_columns)

        # Converting non-numeric columns to numeric where possible
        for col in non_numeric_columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        # Dropping rows with any remaining non-numeric values
        data.dropna(inplace=True)
        print("Data cleaning completed.")

        # Preparing data for model training
        X = data.drop(["PCOS (Y/N)", "Sl. No", "Patient File No.", "Marraige Status (Yrs)", "Blood Group","II    beta-HCG(mIU/mL)","TSH (mIU/L)","Waist:Hip Ratio"], axis=1)
        y = data["PCOS (Y/N)"]

        # Splitting the data into test and training sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        print("Data split into training and test sets.")

        # Fitting the RandomForestClassifier to the training set
        rfc = RandomForestClassifier()
        rfc.fit(X_train, y_train)
        print("Model training completed.")

        # Convert the user input to a NumPy array and reshape it to be 2D
        user_input_reshaped = np.array(user_input).reshape(1, -1)
   
        # Ensure the input data has the correct shape
        if user_input_reshaped.shape[1] != X.shape[1]:
            raise ValueError(f"Expected input shape ({1}, {X.shape[1]}), but got {user_input_reshaped.shape}")

        # Convert user input to DataFrame with feature names
        user_input_df = pd.DataFrame(user_input_reshaped, columns=X.columns)

        # Assuming the scaler used during training
        scaler = StandardScaler()  # Normally you'd load a fitted scaler

        # For demonstration, let's assume no scaling (remove if you're scaling the input)
        user_input_scaled = user_input_df  # Remove this if you're applying scaling

        # Get the probability of PCOS
        probabilities = rfc.predict_proba(user_input_scaled)

        # Extract probability for PCOS (class 1)
        probability_pcos = probabilities[0][1]  # Probability of PCOS (class 1)

        return probability_pcos

if __name__ == "__main__":        
    webServer = HTTPServer((hostName, serverPort), MyServer)
    print("Server started http://%s:%s" % (hostName, serverPort))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")