Training

```
python main.py --model_name marcodiff --run_name mc --batch_size 4 --max_iter 100000 --test_id 9 --save_freq 5000
```

Test

```
python main.py --model_name marcodiff --run_name mc --batch_size 4 --max_iter 100000 --test_batch_size 1 --test_id 9 --save_freq 2500 --mode test --test_iter 100000
```

Constructed LDCT Datasets from NDCT

```
datasets/simulation.py
```
