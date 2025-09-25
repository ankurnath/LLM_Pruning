# SIZE CONSTRAINT EXPERIMENTS

# MAXCOV_SIZE
# python finetune.py --problem "Maximum Coverage" --dataset Facebook
# python finetune.py --problem "Maximum Coverage" --dataset Wiki
# python finetune.py --problem "Maximum Coverage" --dataset Deezer
# python finetune.py --problem "Maximum Coverage" --dataset Slashdot
# python finetune.py --problem "Maximum Coverage" --dataset Twitter
# python finetune.py --problem "Maximum Coverage" --dataset DBLP
# python finetune.py --problem "Maximum Coverage" --dataset YouTube
# python finetune.py --problem "Maximum Coverage" --dataset Skitter

# python test.py --problem "Maximum Coverage" --train_dataset Facebook --test_dataset Facebook --k 500 
# python test.py --problem "Maximum Coverage" --train_dataset Wiki --test_dataset Wiki --k 500 
# python test.py --problem "Maximum Coverage" --train_dataset Deezer --test_dataset Deezer --k 500 
# python test.py --problem "Maximum Coverage" --train_dataset Slashdot --test_dataset Slashdot --k 500 
# python test.py --problem "Maximum Coverage" --train_dataset Twitter --test_dataset Twitter --k 1000 
# python test.py --problem "Maximum Coverage" --train_dataset DBLP --test_dataset DBLP    --k 5000 
# python test.py --problem "Maximum Coverage" --train_dataset YouTube --test_dataset YouTube --k 5000 
# python test.py --problem "Maximum Coverage" --train_dataset Skitter --test_dataset Skitter --k 5000 

# MAXCUT_SIZE
# python finetune.py --problem "Maximum Cut" --dataset Facebook
# python finetune.py --problem "Maximum Cut" --dataset Wiki
# python finetune.py --problem "Maximum Cut" --dataset Deezer
# python finetune.py --problem "Maximum Cut" --dataset Slashdot
# python finetune.py --problem "Maximum Cut" --dataset Twitter
# python finetune.py --problem "Maximum Cut" --dataset DBLP
# python finetune.py --problem "Maximum Cut" --dataset YouTube
# python finetune.py --problem "Maximum Cut" --dataset Skitter
# wait



# python test.py --problem "Maximum Cut" --train_dataset Facebook --test_dataset Facebook --k 500 
# python test.py --problem "Maximum Cut" --train_dataset Wiki --test_dataset Wiki --k 500 
# python test.py --problem "Maximum Cut" --train_dataset Deezer --test_dataset Deezer --k 500 
# python test.py --problem "Maximum Cut" --train_dataset Slashdot --test_dataset Slashdot --k 500 
# python test.py --problem "Maximum Cut" --train_dataset Twitter --test_dataset Twitter --k 1000 
# python test.py --problem "Maximum Cut" --train_dataset DBLP --test_dataset DBLP    --k 5000 
# python test.py --problem "Maximum Cut" --train_dataset YouTube --test_dataset YouTube --k 5000 
# python test.py --problem "Maximum Cut" --train_dataset Skitter --test_dataset Skitter --k 5000 

# IM_SIZE
# python finetune.py --problem "Influence Maximization" --dataset Facebook
# python finetune.py --problem "Influence Maximization" --dataset Wiki
# python finetune.py --problem "Influence Maximization" --dataset Deezer
# python finetune.py --problem "Influence Maximization" --dataset Slashdot
# python finetune.py --problem "Influence Maximization" --dataset Twitter
# python finetune.py --problem "Influence Maximization" --dataset DBLP
# python finetune.py --problem "Influence Maximization" --dataset YouTube
# python finetune.py --problem "Influence Maximization" --dataset Skitter


# python test.py --problem "Influence Maximization" --train_dataset Facebook --test_dataset Facebook --k 500 
# python test.py --problem "Influence Maximization" --train_dataset Wiki --test_dataset Wiki --k 500 
# python test.py --problem "Influence Maximization" --train_dataset Deezer --test_dataset Deezer --k 500 
# python test.py --problem "Influence Maximization" --train_dataset Slashdot --test_dataset Slashdot --k 500 
# python test.py --problem "Influence Maximization" --train_dataset Twitter --test_dataset Twitter --k 1000 
# python test.py --problem "Influence Maximization" --train_dataset DBLP --test_dataset DBLP    --k 5000 
# python test.py --problem "Influence Maximization" --train_dataset YouTube --test_dataset YouTube --k 5000 
# python test.py --problem "Influence Maximization" --train_dataset Skitter --test_dataset Skitter --k 5000





# WEIGHT CONSTRAINT EXPERIMENTS

# MAXCOV_WEIGHT

# python finetune.py --problem "Maximum Coverage Weighted" --dataset Facebook &
# python finetune.py --problem "Maximum Coverage Weighted" --dataset Wiki &
# python finetune.py --problem "Maximum Coverage Weighted" --dataset Deezer &
# python finetune.py --problem "Maximum Coverage Weighted" --dataset Slashdot &
# python finetune.py --problem "Maximum Coverage Weighted" --dataset Twitter &
# python finetune.py --problem "Maximum Coverage Weighted" --dataset DBLP &
# python finetune.py --problem "Maximum Coverage Weighted" --dataset YouTube &
# python finetune.py --problem "Maximum Coverage Weighted" --dataset Skitter &
# wait

# python test.py --problem "Maximum Coverage Weighted" --train_dataset Facebook --test_dataset Facebook --k 500 &
# python test.py --problem "Maximum Coverage Weighted" --train_dataset Wiki --test_dataset Wiki --k 500 &
# python test.py --problem "Maximum Coverage Weighted" --train_dataset Deezer --test_dataset Deezer --k 500 &
# python test.py --problem "Maximum Coverage Weighted" --train_dataset Slashdot --test_dataset Slashdot --k 500 &
# python test.py --problem "Maximum Coverage Weighted" --train_dataset Twitter --test_dataset Twitter --k 1000 &
# python test.py --problem "Maximum Coverage Weighted" --train_dataset DBLP --test_dataset DBLP    --k 5000 &
# python test.py --problem "Maximum Coverage Weighted" --train_dataset YouTube --test_dataset YouTube --k 5000 &
# python test.py --problem "Maximum Coverage Weighted" --train_dataset Skitter --test_dataset Skitter --k 5000 &
# wait

# MAXCUT_WEIGHT

python finetune.py --problem "Maximum Cut Weighted" --dataset Facebook &
python finetune.py --problem "Maximum Cut Weighted" --dataset Wiki &
python finetune.py --problem "Maximum Cut Weighted" --dataset Deezer &
python finetune.py --problem "Maximum Cut Weighted" --dataset Slashdot &
python finetune.py --problem "Maximum Cut Weighted" --dataset Twitter &
python finetune.py --problem "Maximum Cut Weighted" --dataset DBLP &
python finetune.py --problem "Maximum Cut Weighted" --dataset YouTube &
python finetune.py --problem "Maximum Cut Weighted" --dataset Skitter &
wait

python test.py --problem "Maximum Cut Weighted" --train_dataset Facebook --test_dataset Facebook --k 500 &
python test.py --problem "Maximum Cut Weighted" --train_dataset Wiki --test_dataset Wiki --k 500 &
python test.py --problem "Maximum Cut Weighted" --train_dataset Deezer --test_dataset Deezer --k 500 &
python test.py --problem "Maximum Cut Weighted" --train_dataset Slashdot --test_dataset Slashdot --k 500 &
python test.py --problem "Maximum Cut Weighted" --train_dataset Twitter --test_dataset Twitter --k 1000 &
python test.py --problem "Maximum Cut Weighted" --train_dataset DBLP --test_dataset DBLP    --k 5000 &
python test.py --problem "Maximum Cut Weighted" --train_dataset YouTube --test_dataset YouTube --k 5000 &
python test.py --problem "Maximum Cut Weighted" --train_dataset Skitter --test_dataset Skitter --k 5000 &
wait


# IM_WEIGHT
# python finetune.py --problem "Influence Maximization Weighted" --dataset Facebook &
# python finetune.py --problem "Influence Maximization Weighted" --dataset Wiki &
# python finetune.py --problem "Influence Maximization Weighted" --dataset Deezer &
# python finetune.py --problem "Influence Maximization Weighted" --dataset Slashdot &
# python finetune.py --problem "Influence Maximization Weighted" --dataset Twitter &
# python finetune.py --problem "Influence Maximization Weighted" --dataset DBLP &
# python finetune.py --problem "Influence Maximization Weighted" --dataset YouTube &
# python finetune.py --problem "Influence Maximization Weighted" --dataset Skitter &
# wait

# python test.py --problem "Influence Maximization Weighted" --train_dataset Facebook --test_dataset Facebook --k 500 &
# python test.py --problem "Influence Maximization Weighted" --train_dataset Wiki --test_dataset Wiki --k 500 &
# python test.py --problem "Influence Maximization Weighted" --train_dataset Deezer --test_dataset Deezer --k 500 &
# python test.py --problem "Influence Maximization Weighted" --train_dataset Slashdot --test_dataset Slashdot --k 500 &
# python test.py --problem "Influence Maximization Weighted" --train_dataset Twitter --test_dataset Twitter --k 1000 &
# python test.py --problem "Influence Maximization Weighted" --train_dataset DBLP --test_dataset DBLP    --k 5000 &
# python test.py --problem "Influence Maximization Weighted" --train_dataset YouTube --test_dataset YouTube --k 5000 &
# python test.py --problem "Influence Maximization Weighted" --train_dataset Skitter --test_dataset Skitter --k 5000 &
# wait
