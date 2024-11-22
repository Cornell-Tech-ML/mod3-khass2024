# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

## Performance Comparison: GPU vs Fast

The following graph illustrates the speed-ups on large matrix operations that CUDA matrix multiplication has over naive operations.

![Performance graph: GPU vs Fast](graphs/gpu_vs_fast.png)

## Training times

### Simple Dataset w/ Hidden Size 100

Click below for the CPU and GPU times.

<details>
<summary>CPU</summary>
Epoch   0 | loss 7.129110 | correct 43 | Time/epoch 21.874s<br>
Epoch  10 | loss 2.946556 | correct 48 | Time/epoch 0.099s<br>
Epoch  20 | loss 1.897210 | correct 49 | Time/epoch 0.104s<br>
Epoch  30 | loss 2.321437 | correct 49 | Time/epoch 0.101s<br>
Epoch  40 | loss 0.458984 | correct 49 | Time/epoch 0.099s<br>
Epoch  50 | loss 0.310600 | correct 49 | Time/epoch 0.173s<br>
Epoch  60 | loss 1.266586 | correct 50 | Time/epoch 0.099s<br>
Epoch  70 | loss 0.897634 | correct 49 | Time/epoch 0.098s<br>
Epoch  80 | loss 0.249812 | correct 49 | Time/epoch 0.095s<br>
Epoch  90 | loss 0.018999 | correct 50 | Time/epoch 0.096s<br>
Epoch 100 | loss 0.785954 | correct 50 | Time/epoch 0.096s<br>
Epoch 110 | loss 0.445344 | correct 50 | Time/epoch 0.096s<br>
Epoch 120 | loss 0.029815 | correct 50 | Time/epoch 0.095s<br>
Epoch 130 | loss 1.407645 | correct 50 | Time/epoch 0.109s<br>
Epoch 140 | loss 0.729497 | correct 50 | Time/epoch 0.102s<br>
Epoch 150 | loss 0.092497 | correct 50 | Time/epoch 0.099s<br>
Epoch 160 | loss 0.704759 | correct 50 | Time/epoch 0.096s<br>
Epoch 170 | loss 0.332510 | correct 49 | Time/epoch 0.221s<br>
Epoch 180 | loss 0.488567 | correct 50 | Time/epoch 0.097s<br>
Epoch 190 | loss 0.005283 | correct 50 | Time/epoch 0.097s<br>
Epoch 200 | loss 0.822763 | correct 50 | Time/epoch 0.101s<br>
Epoch 210 | loss 0.337094 | correct 49 | Time/epoch 0.098s<br>
Epoch 220 | loss 0.145768 | correct 50 | Time/epoch 0.099s<br>
Epoch 230 | loss 0.111058 | correct 50 | Time/epoch 0.109s<br>
Epoch 240 | loss 0.245131 | correct 50 | Time/epoch 0.097s<br>
Epoch 250 | loss 0.008469 | correct 50 | Time/epoch 0.096s<br>
Epoch 260 | loss 0.261438 | correct 50 | Time/epoch 0.097s<br>
Epoch 270 | loss 0.978733 | correct 50 | Time/epoch 0.096s<br>
Epoch 280 | loss 0.006038 | correct 49 | Time/epoch 0.098s<br>
Epoch 290 | loss 0.126511 | correct 50 | Time/epoch 0.228s<br>
Epoch 300 | loss 0.561839 | correct 50 | Time/epoch 0.098s<br>
Epoch 310 | loss 0.498986 | correct 50 | Time/epoch 0.098s<br>
Epoch 320 | loss 0.009375 | correct 50 | Time/epoch 0.096s<br>
Epoch 330 | loss 0.008794 | correct 50 | Time/epoch 0.105s<br>
Epoch 340 | loss 0.276903 | correct 50 | Time/epoch 0.099s<br>
Epoch 350 | loss 0.450139 | correct 50 | Time/epoch 0.097s<br>
Epoch 360 | loss 0.592408 | correct 50 | Time/epoch 0.097s<br>
Epoch 370 | loss 0.569468 | correct 50 | Time/epoch 0.096s<br>
Epoch 380 | loss 0.708758 | correct 50 | Time/epoch 0.095s<br>
Epoch 390 | loss 0.707219 | correct 50 | Time/epoch 0.116s<br>
Epoch 400 | loss 0.634800 | correct 50 | Time/epoch 0.097s<br>
Epoch 410 | loss 0.123268 | correct 50 | Time/epoch 0.223s<br>
Epoch 420 | loss 0.916432 | correct 50 | Time/epoch 0.225s<br>
Epoch 430 | loss 0.000077 | correct 50 | Time/epoch 0.110s<br>
Epoch 440 | loss 0.059406 | correct 50 | Time/epoch 0.100s<br>
Epoch 450 | loss 0.333143 | correct 50 | Time/epoch 0.096s<br>
Epoch 460 | loss 0.160591 | correct 50 | Time/epoch 0.096s<br>
Epoch 470 | loss 0.001260 | correct 50 | Time/epoch 0.097s<br>
Epoch 480 | loss 0.253075 | correct 50 | Time/epoch 0.099s<br>
Epoch 490 | loss 0.276514 | correct 50 | Time/epoch 0.096s<br>
</details>

<details>
<summary>GPU</summary>
Epoch   0 | loss 5.157476 | correct 46 | Time/epoch 4.484s<br>
Epoch  10 | loss 1.250286 | correct 48 | Time/epoch 1.376s<br>
Epoch  20 | loss 1.285082 | correct 49 | Time/epoch 1.405s<br>
Epoch  30 | loss 0.522122 | correct 49 | Time/epoch 1.400s<br>
Epoch  40 | loss 0.725272 | correct 50 | Time/epoch 1.383s<br>
Epoch  50 | loss 0.262390 | correct 49 | Time/epoch 1.841s<br>
Epoch  60 | loss 0.403588 | correct 50 | Time/epoch 1.372s<br>
Epoch  70 | loss 1.018233 | correct 49 | Time/epoch 1.378s<br>
Epoch  80 | loss 0.561936 | correct 50 | Time/epoch 1.380s<br>
Epoch  90 | loss 0.601933 | correct 49 | Time/epoch 1.399s<br>
Epoch 100 | loss 0.421189 | correct 50 | Time/epoch 1.718s<br>
Epoch 110 | loss 0.582415 | correct 49 | Time/epoch 1.456s<br>
Epoch 120 | loss 1.639665 | correct 49 | Time/epoch 1.385s<br>
Epoch 130 | loss 0.105992 | correct 49 | Time/epoch 1.390s<br>
Epoch 140 | loss 0.269902 | correct 49 | Time/epoch 1.373s<br>
Epoch 150 | loss 0.101179 | correct 49 | Time/epoch 1.373s<br>
Epoch 160 | loss 0.938684 | correct 49 | Time/epoch 2.012s<br>
Epoch 170 | loss 0.123711 | correct 50 | Time/epoch 1.385s<br>
Epoch 180 | loss 0.020657 | correct 50 | Time/epoch 1.374s<br>
Epoch 190 | loss 0.017975 | correct 49 | Time/epoch 1.362s<br>
Epoch 200 | loss 1.527343 | correct 49 | Time/epoch 1.402s<br>
Epoch 210 | loss 0.303688 | correct 49 | Time/epoch 1.878s<br>
Epoch 220 | loss 0.092926 | correct 49 | Time/epoch 1.370s<br>
Epoch 230 | loss 0.028534 | correct 49 | Time/epoch 1.377s<br>
Epoch 240 | loss 0.211879 | correct 50 | Time/epoch 1.374s<br>
Epoch 250 | loss 1.598304 | correct 49 | Time/epoch 1.365s<br>
Epoch 260 | loss 0.037128 | correct 50 | Time/epoch 1.360s<br>
Epoch 270 | loss 0.122906 | correct 49 | Time/epoch 2.052s<br>
Epoch 280 | loss 0.106173 | correct 49 | Time/epoch 1.371s<br>
Epoch 290 | loss 1.020686 | correct 50 | Time/epoch 1.365s<br>
Epoch 300 | loss 0.061129 | correct 50 | Time/epoch 1.386s<br>
Epoch 310 | loss 0.553587 | correct 50 | Time/epoch 1.367s<br>
Epoch 320 | loss 0.025663 | correct 50 | Time/epoch 1.366s<br>
Epoch 330 | loss 0.305075 | correct 50 | Time/epoch 2.015s<br>
Epoch 340 | loss 0.036088 | correct 50 | Time/epoch 1.414s<br>
Epoch 350 | loss 0.020005 | correct 50 | Time/epoch 1.434s<br>
Epoch 360 | loss 0.047750 | correct 49 | Time/epoch 1.362s<br>
Epoch 370 | loss 0.024887 | correct 50 | Time/epoch 1.436s<br>
Epoch 380 | loss 0.436363 | correct 49 | Time/epoch 1.776s<br>
Epoch 390 | loss 0.024866 | correct 50 | Time/epoch 1.596s<br>
Epoch 400 | loss 0.154126 | correct 49 | Time/epoch 1.443s<br>
Epoch 410 | loss 0.010351 | correct 49 | Time/epoch 1.426s<br>
Epoch 420 | loss 0.392520 | correct 50 | Time/epoch 1.445s<br>
Epoch 430 | loss 0.957785 | correct 50 | Time/epoch 1.423s<br>
Epoch 440 | loss 0.134028 | correct 49 | Time/epoch 1.979s<br>
Epoch 450 | loss 0.009593 | correct 50 | Time/epoch 1.434s<br>
Epoch 460 | loss 0.699601 | correct 49 | Time/epoch 1.424s<br>
Epoch 470 | loss 0.711810 | correct 49 | Time/epoch 1.436s<br>
Epoch 480 | loss 1.030740 | correct 50 | Time/epoch 1.454s<br>
Epoch 490 | loss 0.205119 | correct 50 | Time/epoch 1.430s<br>
</details>

### Split Dataset w/ Hidden Size 100

Click below for the CPU and GPU times.

<details>
<summary>CPU</summary>
Epoch   0 | loss 5.513937 | correct 36 | Time/epoch 21.815s<br>
Epoch  10 | loss 4.211195 | correct 41 | Time/epoch 0.097s<br>
Epoch  20 | loss 3.511429 | correct 39 | Time/epoch 0.097s<br>
Epoch  30 | loss 3.301197 | correct 46 | Time/epoch 0.096s<br>
Epoch  40 | loss 0.660914 | correct 34 | Time/epoch 0.095s<br>
Epoch  50 | loss 1.865769 | correct 48 | Time/epoch 0.095s<br>
Epoch  60 | loss 2.660754 | correct 48 | Time/epoch 0.097s<br>
Epoch  70 | loss 3.504392 | correct 47 | Time/epoch 0.096s<br>
Epoch  80 | loss 2.096482 | correct 46 | Time/epoch 0.106s<br>
Epoch  90 | loss 0.794168 | correct 48 | Time/epoch 0.096s<br>
Epoch 100 | loss 3.066883 | correct 48 | Time/epoch 0.095s<br>
Epoch 110 | loss 1.679946 | correct 45 | Time/epoch 0.147s<br>
Epoch 120 | loss 2.430616 | correct 46 | Time/epoch 0.184s<br>
Epoch 130 | loss 1.929402 | correct 50 | Time/epoch 0.095s<br>
Epoch 140 | loss 1.217379 | correct 50 | Time/epoch 0.095s<br>
Epoch 150 | loss 1.196383 | correct 49 | Time/epoch 0.095s<br>
Epoch 160 | loss 1.352175 | correct 50 | Time/epoch 0.096s<br>
Epoch 170 | loss 2.091854 | correct 46 | Time/epoch 0.095s<br>
Epoch 180 | loss 1.072992 | correct 50 | Time/epoch 0.095s<br>
Epoch 190 | loss 1.029824 | correct 50 | Time/epoch 0.094s<br>
Epoch 200 | loss 0.615876 | correct 49 | Time/epoch 0.095s<br>
Epoch 210 | loss 1.028453 | correct 50 | Time/epoch 0.098s<br>
Epoch 220 | loss 1.207640 | correct 50 | Time/epoch 0.095s<br>
Epoch 230 | loss 0.461682 | correct 50 | Time/epoch 0.095s<br>
Epoch 240 | loss 1.264002 | correct 50 | Time/epoch 0.131s<br>
Epoch 250 | loss 0.634347 | correct 50 | Time/epoch 0.094s<br>
Epoch 260 | loss 0.277020 | correct 50 | Time/epoch 0.107s<br>
Epoch 270 | loss 1.721210 | correct 45 | Time/epoch 0.098s<br>
Epoch 280 | loss 1.586376 | correct 48 | Time/epoch 0.095s<br>
Epoch 290 | loss 0.191044 | correct 50 | Time/epoch 0.094s<br>
Epoch 300 | loss 1.263995 | correct 49 | Time/epoch 0.095s<br>
Epoch 310 | loss 0.205815 | correct 50 | Time/epoch 0.095s<br>
Epoch 320 | loss 0.536306 | correct 50 | Time/epoch 0.096s<br>
Epoch 330 | loss 0.874386 | correct 50 | Time/epoch 0.095s<br>
Epoch 340 | loss 0.929821 | correct 50 | Time/epoch 0.104s<br>
Epoch 350 | loss 0.331829 | correct 50 | Time/epoch 0.094s<br>
Epoch 360 | loss 0.743873 | correct 50 | Time/epoch 0.148s<br>
Epoch 370 | loss 0.033729 | correct 49 | Time/epoch 0.164s<br>
Epoch 380 | loss 0.313251 | correct 50 | Time/epoch 0.094s<br>
Epoch 390 | loss 0.194911 | correct 50 | Time/epoch 0.097s<br>
Epoch 400 | loss 0.206180 | correct 50 | Time/epoch 0.094s<br>
Epoch 410 | loss 0.615259 | correct 50 | Time/epoch 0.095s<br>
Epoch 420 | loss 0.041167 | correct 50 | Time/epoch 0.095s<br>
Epoch 430 | loss 0.302820 | correct 50 | Time/epoch 0.107s<br>
Epoch 440 | loss 0.227690 | correct 50 | Time/epoch 0.094s<br>
Epoch 450 | loss 0.348787 | correct 50 | Time/epoch 0.094s<br>
Epoch 460 | loss 0.294791 | correct 50 | Time/epoch 0.094s<br>
Epoch 470 | loss 0.245370 | correct 50 | Time/epoch 0.098s<br>
Epoch 480 | loss 0.256506 | correct 50 | Time/epoch 0.095s<br>
Epoch 490 | loss 0.363866 | correct 50 | Time/epoch 0.213s<br>
</details>

<details>
<summary>GPU</summary>
Epoch   0 | loss 6.360098 | correct 28 | Time/epoch 5.434s<br>
Epoch  10 | loss 5.562011 | correct 39 | Time/epoch 1.387s<br>
Epoch  20 | loss 6.310424 | correct 37 | Time/epoch 1.396s<br>
Epoch  30 | loss 4.915457 | correct 45 | Time/epoch 1.395s<br>
Epoch  40 | loss 3.011965 | correct 45 | Time/epoch 1.704s<br>
Epoch  50 | loss 3.513882 | correct 46 | Time/epoch 1.423s<br>
Epoch  60 | loss 2.493221 | correct 47 | Time/epoch 1.400s<br>
Epoch  70 | loss 1.650403 | correct 47 | Time/epoch 1.393s<br>
Epoch  80 | loss 2.706538 | correct 46 | Time/epoch 1.383s<br>
Epoch  90 | loss 2.473138 | correct 48 | Time/epoch 1.502s<br>
Epoch 100 | loss 1.113997 | correct 49 | Time/epoch 1.690s<br>
Epoch 110 | loss 2.056636 | correct 49 | Time/epoch 1.394s<br>
Epoch 120 | loss 1.235082 | correct 49 | Time/epoch 1.392s<br>
Epoch 130 | loss 2.180583 | correct 48 | Time/epoch 1.396s<br>
Epoch 140 | loss 1.861788 | correct 48 | Time/epoch 1.601s<br>
Epoch 150 | loss 0.697359 | correct 48 | Time/epoch 1.589s<br>
Epoch 160 | loss 0.508582 | correct 49 | Time/epoch 1.382s<br>
Epoch 170 | loss 1.546126 | correct 50 | Time/epoch 1.402s<br>
Epoch 180 | loss 0.710645 | correct 49 | Time/epoch 1.389s<br>
Epoch 190 | loss 2.128681 | correct 48 | Time/epoch 1.393s<br>
Epoch 200 | loss 1.128437 | correct 49 | Time/epoch 1.930s<br>
Epoch 210 | loss 0.784599 | correct 49 | Time/epoch 1.386s<br>
Epoch 220 | loss 0.082554 | correct 49 | Time/epoch 1.415s<br>
Epoch 230 | loss 1.076027 | correct 50 | Time/epoch 1.386s<br>
Epoch 240 | loss 1.637862 | correct 48 | Time/epoch 1.381s<br>
Epoch 250 | loss 2.266303 | correct 50 | Time/epoch 1.939s<br>
Epoch 260 | loss 0.397726 | correct 49 | Time/epoch 1.405s<br>
Epoch 270 | loss 0.491331 | correct 49 | Time/epoch 1.409s<br>
Epoch 280 | loss 0.738094 | correct 50 | Time/epoch 1.992s<br>
Epoch 290 | loss 0.487938 | correct 50 | Time/epoch 1.377s<br>
Epoch 300 | loss 1.193812 | correct 49 | Time/epoch 2.049s<br>
Epoch 310 | loss 1.200962 | correct 50 | Time/epoch 1.376s<br>
Epoch 320 | loss 0.316224 | correct 49 | Time/epoch 1.378s<br>
Epoch 330 | loss 0.302942 | correct 49 | Time/epoch 1.388s<br>
Epoch 340 | loss 0.501311 | correct 50 | Time/epoch 1.383s<br>
Epoch 350 | loss 0.627615 | correct 50 | Time/epoch 1.728s<br>
Epoch 360 | loss 1.266220 | correct 50 | Time/epoch 1.503s<br>
Epoch 370 | loss 0.702427 | correct 49 | Time/epoch 1.451s<br>
Epoch 380 | loss 0.984904 | correct 49 | Time/epoch 1.457s<br>
Epoch 390 | loss 1.331533 | correct 50 | Time/epoch 1.460s<br>
Epoch 400 | loss 0.263109 | correct 49 | Time/epoch 1.451s<br>
Epoch 410 | loss 0.132350 | correct 50 | Time/epoch 1.891s<br>
Epoch 420 | loss 0.315379 | correct 50 | Time/epoch 1.473s<br>
Epoch 430 | loss 0.865725 | correct 50 | Time/epoch 1.448s<br>
Epoch 440 | loss 0.253330 | correct 49 | Time/epoch 1.450s<br>
Epoch 450 | loss 0.036919 | correct 49 | Time/epoch 1.510s<br>
Epoch 460 | loss 0.557020 | correct 49 | Time/epoch 1.849s<br>
Epoch 470 | loss 0.792600 | correct 50 | Time/epoch 1.446s<br>
Epoch 480 | loss 0.848296 | correct 49 | Time/epoch 1.449s<br>
Epoch 490 | loss 0.019554 | correct 49 | Time/epoch 1.447s<br>
</details>

### Xor Dataset w/ Hidden Size 100

Click below for the CPU and GPU times.

<details>
<summary>CPU</summary>
</details>

<details>
<summary>GPU</summary>
</details>

### Simple Dataset w/ Hidden Size 200

Click below for the CPU and GPU times.

<details>
<summary>CPU</summary>
</details>

<details>
<summary>GPU</summary>
</details>