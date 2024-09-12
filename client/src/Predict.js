import { useState } from "react";
import axios from "axios";

const Predict = () => {
  const [file, setFile] = useState();
  const [response, setResponse] = useState("");

  const handleChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleClick = () => {
    if (file) {
      const formData = new FormData();
      formData.append("file", file);
      console.log(file);

      axios
        .post("http://127.0.0.1:5000/", formData, {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        })
        .then((res) => {
          console.log(res.data);
          setResponse(res.data.predicted_class);
        })
        .catch((err) => {
          console.log(err);
        });
    } else {
      console.log("No file selected");
    }
  };

  return (
    <div className="mx-auto p-6 rounded-lg shadow-md bg-gray-800 flex justify-center h-screen items-center flex-col">
      <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">
        Upload and Predict
      </h2>
      <label
        className="block text-sm font-medium text-gray-900 dark:text-gray-300 mb-2"
        htmlFor="file_input"
      >
        Select an image to upload
      </label>
      <input
        className="block w-96 text-sm text-gray-900 border border-gray-300 rounded-lg cursor-pointer bg-gray-50 dark:text-gray-400 focus:outline-none dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400"
        aria-describedby="file"
        id="file"
        type="file"
        accept="image/*"
        name="file"
        onChange={handleChange}
      />
      <p
        className="mt-1 text-sm text-gray-500 dark:text-gray-300"
        id="file_input_help"
      >
        SVG, PNG, JPG or GIF (MAX. 800x400px).
      </p>
      {file && (
        <div className="flex justify-center flex-col">
          <img
            className="mt-4 w-96 h-96 object-cover"
            src={URL.createObjectURL(file)}
            alt="uploaded_image"
          />
          <button
            className="mt-4 w-50 bg-blue-600 hover:bg-blue-500 text-white font-bold py-2 px-4 rounded-lg transition-colors"
            onClick={handleClick}
          >
            Predict
          </button>
        </div>
      )}
      {response && (
        <div className="mt-4 w-full text-white text-center font-bold py-2 px-4 rounded-lg">
          Predicted Class: {response}
        </div>
      )}
    </div>
  );
};

export default Predict;
