Elementary implementation of [LLMFactor](https://aclanthology.org/2024.findings-acl.185/) and attempt of adversarial attack on it.

## How to run
1. Install the requirements
    ```bash
    python -m venv {any_name}
    pip install -r requirements.txt
    ```
    
    Or, use with conda environment
    ```bash
    conda env create -n {any_name} python=3.12
    conda activate {any_name}
    pip install -r requirements.txt
    ```
   
2. Make `config.json`

   ```
   cp config.json.example config.json
   ```

3. Run the code
    ```bash
    python -m llmfactor
    ```

    Also, you can specify the config file path
    
    ```bash
    python -m llmfactor --config-file {config_file_path}
    ```