# GEMINI Project Analysis: Stock Scanner & Filter App

## Project Overview

This project is a powerful stock scanning and filtering application built with Streamlit. It is designed to be similar to Chartink, allowing users to upload stock data, apply technical indicators, and create custom filters to identify trading opportunities. The application is optimized for performance, with features like caching, vectorized operations, and memory-efficient data processing.

**Key Technologies:**

*   **Backend:** Python
*   **Frontend:** Streamlit
*   **Data Manipulation:** Pandas, NumPy
*   **Charting:** Plotly
*   **File Formats:** CSV, Excel, Parquet

**Architecture:**

The application follows a modular structure, with separate files for the main application logic, filter engine, technical indicators, UI components, and utility functions. This separation of concerns makes the codebase easier to understand, maintain, and extend.

## Building and Running

**1. Install Dependencies:**

To install the required Python packages, run the following command:

```bash
pip install -r requirements_file.txt
```

**2. Run the Application:**

To start the Streamlit application, execute the following command:

```bash
streamlit run stock_scanner_main.py
```

The application will then be available in your web browser at `http://localhost:8501`.

**3. Running Tests:**

While no explicit test command is specified, the project uses `pytest` for testing. To run the tests, you can use the following command:

```bash
pytest
```

## Development Conventions

**Coding Style:**

The codebase appears to follow the PEP 8 style guide for Python. It is recommended to use a linter like `flake8` and a code formatter like `black` to maintain consistency.

**Testing:**

The project includes a suite of tests using the `pytest` framework. All new features and bug fixes should be accompanied by corresponding tests to ensure the stability and reliability of the application.

**Contributions:**

Contributions to the project are welcome. To contribute, please follow these steps:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and add or update tests as needed.
4.  Ensure that all tests pass.
5.  Submit a pull request with a clear description of your changes.
