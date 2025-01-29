# Poetry


Follow the below steps from a Vertex AI Workbench terminal. See [documentation](https://python-poetry.org/docs/dependency-specification/#caret-requirements) for details


## 1. create new conda environment


```
export ENV_NAME=py310_tf211_v1
export ENV_DISPLAY=py310_tf211_v1

echo $ENV_NAME
echo $ENV_DISPLAY

conda create -n $ENV_NAME -y python=3.10.16
conda activate $ENV_NAME
conda install pip
pip install -U poetry ipykernel packaging
```

*expand for more...*

<details>
  <summary>add conda env as notebook kernel</summary>

Add custom `conda` environment as a [kernel](https://jupyterlab.readthedocs.io/en/stable/user/documents_kernels.html) for Vertex AI Workbench instance(s)

</details>


## 2. interactive terminal


```
export DIR_NAME=poetry_dir
mkdir $DIR_NAME
cd $DIR_NAME
```

initialize `poetry` to begin an interactive shell 
* For each line, press enter to return blank or default values 
* `Author` is the only config that cannot be left blank 

```
poetry init

## begin interactive output

Package name [fed_towers]:  
Version [0.1.0]:  
Description []:  
Author [None, n to skip]: jtott
License []:  
Compatible Python versions [>=3.10]: 
```


## 3. Update `pyproject.toml`


1. Open `pyproject.toml`

2. Find this line: `requires-python = ">=3.10"` 

3. Add a max version to create a range: `requires-python = ">=3.10, <3.13"`



## 4. install dependencies


See how poetry handles no version specs
* tfrs == 0.7.2
* tf == 2.11.0

```
poetry add tensorflow[and-cuda]==2.11.0
poetry add tensorflow-recommenders==0.7.2
poetry add numba
poetry add "google-cloud-aiplatform>=1.60.0"
poetry add pandas
poetry add "numpy@^1.24.0"
poetry add "scann@^1.2.9"
```


## 5. Add conda kernel


```
DL_ANACONDA_ENV_HOME="${DL_ANACONDA_HOME}/envs/$ENV_NAME"
echo $DL_ANACONDA_ENV_HOME

python -m ipykernel install --prefix "${DL_ANACONDA_ENV_HOME}" --name $ENV_NAME --display-name $ENV_DISPLAY
```

After reloading the Workbench instance (`ctrl + r`), you should see `$ENV_DISPLAY` available as a notebook kernel 


# Notes


### version constraints


this command:

```
poetry add tensorflow==2.14.1
```

is the same as: 

```
poetry add tensorflow@2.14.1
```

combine with extras as one might expect: `package[extra]@version`

```
poetry add tensorflow[and-cuda]@2.14.1
```

all these will add the following to your `pyproject.toml`: 

```
dependencies = [
    "tensorflow[and-cuda] (==2.14.1)",
]
```

### inequality vs exact requirements

**Inequality requirements** allow flexibility defining **either a version range** or **exact version** (e.g., `>= 1.2.0`)
* Restrict depndencies (e.g., `!= 1.2.3`)
* Seperate multiple version requirements with comma: `>= 1.2, < 1.5`

**Exact requirements** allow specifying an **exact** `<package>` version (e.g., `tensorflow==2.14.1`)
* Poetry will install this version and this version only
* If other dependencies require a different version, **the solver will ultimately fail** and abort any install or update procedures

**Using the `@` operator**
* When adding dependencies via `poetry add`, we can use the `@` operator similar to `==` syntax
* This also allows prefixing any specifiers that are valid in `pyproject.toml` (see below)

**The caret (`^`) symbol allows compatabile updates to a specified version** 
* With caret, an update is allowed if the new version number does not modify the left-most non-zero digit in the major, minor, patch grouping. 

> If we ran these:
>   1. `poetry add requests@^2.13.0`
  2. `poetry update requests`
  3. poetry updates to version `2.14.0` (if available)
  4. poetry **would not** update to version `3.0.0` bc that would `modify the left-most non-zero digit in the major, minor, patch grouping` 
 
> If instead, we ran these:
>   1. `poetry add requests@^0.1.13`, 
  2. `poetry update requests`
  3. poetry updates to version `0.1.14`
  4. poetry **would not** update to version `0.2.0`

to learn more, check out the [caret requirements](https://python-poetry.org/docs/dependency-specification/#caret-requirements)