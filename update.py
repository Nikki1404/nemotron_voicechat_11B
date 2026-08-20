 => [ 7/13] COPY requirements.txt /app/requirements.txt                                                                                                                                                    0.0s
 => ERROR [ 8/13] RUN python -m pip install       --no-build-isolation       -r /app/requirements.txt                                                                                                      1.8s
------
 > [ 8/13] RUN python -m pip install       --no-build-isolation       -r /app/requirements.txt:
0.776 Collecting transformers==4.56.0 (from -r /app/requirements.txt (line 1))
0.877   Downloading transformers-4.56.0-py3-none-any.whl.metadata (40 kB)
1.111 Collecting tokenizers==0.22.0 (from -r /app/requirements.txt (line 2))
1.124   Downloading tokenizers-0.22.0-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.8 kB)
1.153 Collecting lhotse==1.32.2 (from -r /app/requirements.txt (line 3))
1.175   Downloading lhotse-1.32.2-py3-none-any.whl.metadata (19 kB)
1.220 Collecting huggingface-hub==0.34.4 (from -r /app/requirements.txt (line 4))
1.234   Downloading huggingface_hub-0.34.4-py3-none-any.whl.metadata (14 kB)
1.282 Collecting hf-xet==1.1.9 (from -r /app/requirements.txt (line 5))
1.294   Downloading hf_xet-1.1.9-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.7 kB)
1.325 Collecting torchcodec==0.10.0 (from -r /app/requirements.txt (line 6))
1.339   Downloading torchcodec-0.10.0-cp312-cp312-manylinux_2_28_x86_64.whl.metadata (11 kB)
1.356 Collecting torch_audiomentations (from -r /app/requirements.txt (line 7))
1.369   Downloading torch_audiomentations-0.12.0-py3-none-any.whl.metadata (15 kB)
1.387 Collecting jinja2 (from -r /app/requirements.txt (line 8))
1.399   Downloading jinja2-3.1.6-py3-none-any.whl.metadata (2.9 kB)
1.430 Collecting ninja (from -r /app/requirements.txt (line 10))
1.444   Downloading ninja-1.13.0-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (5.1 kB)
1.445 Requirement already satisfied: packaging in /opt/conda/lib/python3.12/site-packages (from -r /app/requirements.txt (line 11)) (24.2)
1.446 Requirement already satisfied: wheel in /opt/conda/lib/python3.12/site-packages (from -r /app/requirements.txt (line 12)) (0.48.0)
1.460 Collecting einops (from -r /app/requirements.txt (line 13))
1.474   Downloading einops-0.8.2-py3-none-any.whl.metadata (13 kB)
1.500 Collecting causal-conv1d==1.6.2.post1 (from -r /app/requirements.txt (line 15))
1.512   Downloading causal_conv1d-1.6.2.post1.tar.gz (29 kB)
1.526   Preparing metadata (pyproject.toml): started
1.687   Preparing metadata (pyproject.toml): finished with status 'error'
1.692   error: subprocess-exited-with-error
1.692
1.692   × Preparing metadata (pyproject.toml) did not run successfully.
1.692   │ exit code: 1
1.692   ╰─> [18 lines of output]
1.692       /opt/conda/lib/python3.12/site-packages/wheel/bdist_wheel.py:4: FutureWarning: The 'wheel' package is no longer the canonical location of the 'bdist_wheel' command, and will be removed in a future release. Please update to setuptools v70.1 or later which contains an integrated version of this command.
1.692         warn(
1.692       Traceback (most recent call last):
1.692         File "/opt/conda/lib/python3.12/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 389, in <module>
1.692           main()
1.692         File "/opt/conda/lib/python3.12/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 373, in main
1.692           json_out["return_val"] = hook(**hook_input["kwargs"])
1.692                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1.692         File "/opt/conda/lib/python3.12/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 175, in prepare_metadata_for_build_wheel
1.692           return hook(metadata_directory, config_settings)
1.692                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1.692         File "/opt/conda/lib/python3.12/site-packages/setuptools/build_meta.py", line 380, in prepare_metadata_for_build_wheel
1.692           self.run_setup()
1.692         File "/opt/conda/lib/python3.12/site-packages/setuptools/build_meta.py", line 317, in run_setup
1.692           exec(code, locals())  # noqa: S102 # exec is intentional here
1.692           ^^^^^^^^^^^^^^^^^^^^
1.692         File "<string>", line 20, in <module>
1.692       ModuleNotFoundError: No module named 'torch'
1.692       [end of output]
1.692
1.692   note: This error originates from a subprocess, and is likely not a problem with pip.
1.694 error: metadata-generation-failed
1.694
1.694 × Encountered error while generating package metadata.
1.694 ╰─> causal-conv1d
1.694
1.694 note: This is an issue with the package mentioned above, not pip.
1.694 hint: See above for details.
------
ERROR: failed to build: failed to solve: process "/bin/sh -c python -m pip install       --no-build-isolation       -r /app/requirements.txt" did not complete successfully: exit code: 1
