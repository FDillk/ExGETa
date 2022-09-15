# <img src="https://github.com/FDillk/ExGETa/blob/main/tools/ai_logo.png" width="48">     ExGETa

This is a framework for Expandable Generation of Evaluation Tasks developed as a master's thesis at the TU Dortmund University.
The state of this tool at this time is *work in progress* and further research and expansions are needed.

## Description

The framework is based on the generation of evaluation procedures (evaluation tasks) from a database source. The stored evaluation tasks need to be annotated with information about their applicability.

The exact structure for tasks can be explored in the original work, soon made available online. When the thesis has been published, a link to the file will be inserted at this point.

## Quickstart

> The usage of the sample MongoDB database requires an authorized account. This account can then be used to access the existing evaluation tasks or add new tasks to the database. For access please contact me.

To show a quick demonstration of the framework workflow, a trained model on the Ionosphere dataset is available in the *example* folder.
This model is a classification SVM capable of predicting the label with moderately good accuracy.

Executing the framework and storing the evaluation results requires a running [mlflow](https://mlflow.org/) instance with a result storage. For use of a local SQL database running on port 3306, a simple setup with pymsql could look like this:

`mlflow server --backend-store-uri mysql+pymysql://USERNAME:PASSWORD@localhost:3306/DBNAME --default-artifact-root file:/./mlruns -h 0.0.0.0 -p 5000`

After the setup of mlflow, the username and password of the server for evaluation tasks need to be entered in the *dbconnector.py* file. This manual step is currently necessary to ensure the integrity of the task database during grading of the thesis.

The execution of the generation of evaluations using the example files can then be started using the following command:

`python exgeta.py ./example/options.json`

For dependency management, the *conda.yaml* file may be used.

This command results in the generation of a set of evaluation tasks in the newly created subdirectory */ExGETa_ionosphere*. Each of the tasks also comes with a conda file for management of environments. The tasks can then be executed by using the command *python evaluation_task.py* on each individual task.

The results of the evaluations are stored in the result database. Additionally to the normal route of accessing the results using an API, the mlflow UI can also be used. When using the standard setup, this interface can be viewed at http://localhost:5000/.

## Usage

The framework can be used mostly with SVM models at the moment. A JSON file must be supplied, that gives basic information about the model and dataset (see example file as reference). Important elements of this JSON are the locations for model and dataset. The framework can the be executed using:

`python exgeta.py PATH_TO_JSON`

The execution result in the generation of a set of evaluation tasks, that each evaluate different aspects of the model. Only those tasks from the database are generated that are applicable to the inputs at hand.

Each evaluation task can be executed by running:

`python evaluation_task.py`

The conda files can help with execution problems. 
All of the results are logged to the mlflow tracking server and can be viewed in the mlflow UI.

## Further information

Created as part of a master's thesis at the [Chair VIII Artificial Intelligence](https://www-ai.cs.tu-dortmund.de/index.html) at the [TU Dortmund University](https://www.tu-dortmund.de/en/).

The logo was generated using the [Midjourney AI](https://www.midjourney.com/home/).
