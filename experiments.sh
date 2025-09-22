# python main.py --problem "Maximum Coverage" --budget 100 --dataset "HK" --iterations 10
# # python main.py --problem "Maximum Coverage Weighted" --budget 100 --dataset "Wiki" --iterations 10
# # python main.py --problem "Maximum Coverage Weighted" --budget 100 --dataset "Deezer" --iterations 10
# # python main.py --problem "Maximum Coverage Weighted" --budget 100 --dataset "Slashdot" --iterations 10
# # python main.py --problem "Maximum Coverage Weighted" --budget 100 --dataset "Twitter" --iterations 10
# # python main.py --problem "Maximum Coverage Weighted" --budget 100 --dataset "DBLP" --iterations 10
# # python main.py --problem "Maximum Coverage Weighted" --budget 100 --dataset "YouTube" --iterations 10
# # python main.py --problem "Maximum Coverage Weighted" --budget 100 --dataset "Skitter" --iterations 10
# python test.py --problem "Maximum Coverage" --train_dataset HK --test_dataset Facebook
# python test.py --problem "Maximum Coverage" --train_dataset HK --test_dataset Wiki
# python test.py --problem "Maximum Coverage" --train_dataset HK --test_dataset Deezer
# python test.py --problem "Maximum Coverage" --train_dataset HK --test_dataset Slashdot
# python test.py --problem "Maximum Coverage" --train_dataset HK --test_dataset Twitter
# python test.py --problem "Maximum Coverage" --train_dataset HK --test_dataset DBLP
# python test.py --problem "Maximum Coverage" --train_dataset HK --test_dataset YouTube
# python test.py --problem "Maximum Coverage" --train_dataset HK --test_dataset Skitter



# python finetune.py --problem "Influence Maximization" --dataset Facebook
# python finetune.py --problem "Influence Maximization" --dataset Wiki
# python finetune.py --problem "Influence Maximization" --dataset Deezer
# python finetune.py --problem "Influence Maximization" --dataset Slashdot
# python finetune.py --problem "Influence Maximization" --dataset Twitter
# python finetune.py --problem "Influence Maximization" --dataset DBLP
# python finetune.py --problem "Influence Maximization" --dataset YouTube
# python finetune.py --problem "Influence Maximization" --dataset Skitter

# python test.py --problem "Influence Maximization" --train_dataset Facebook --test_dataset Facebook
# python test.py --problem "Influence Maximization" --train_dataset Wiki --test_dataset Wiki
# python test.py --problem "Influence Maximization" --train_dataset Deezer --test_dataset Deezer
# python test.py --problem "Influence Maximization" --train_dataset Slashdot --test_dataset Slashdot
# python test.py --problem "Influence Maximization" --train_dataset Twitter --test_dataset Twitter
# python test.py --problem "Influence Maximization" --train_dataset DBLP --test_dataset DBLP
# python test.py --problem "Influence Maximization" --train_dataset YouTube --test_dataset YouTube
# python test.py --problem "Influence Maximization" --train_dataset Skitter --test_dataset Skitter.


# python test.py --problem "Maximum Cut" --train_dataset HK --test_dataset Facebook
# python test.py --problem "Maximum Cut" --train_dataset HK --test_dataset Wiki
# python test.py --problem "Maximum Cut" --train_dataset HK --test_dataset Deezer
# python test.py --problem "Maximum Cut" --train_dataset HK --test_dataset Slashdot
# python test.py --problem "Maximum Cut" --train_dataset HK --test_dataset Twitter
# python test.py --problem "Maximum Cut" --train_dataset HK --test_dataset DBLP
# python test.py --problem "Maximum Cut" --train_dataset HK --test_dataset YouTube
# python test.py --problem "Maximum Cut" --train_dataset HK --test_dataset Skitter

# python test.py --problem "Influence Maximization" --train_dataset HK --test_dataset Facebook
# python test.py --problem "Influence Maximization" --train_dataset HK --test_dataset Wiki
# python test.py --problem "Influence Maximization" --train_dataset HK --test_dataset Deezer
# python test.py --problem "Influence Maximization" --train_dataset HK --test_dataset Slashdot
# python test.py --problem "Influence Maximization" --train_dataset HK --test_dataset Twitter
# python test.py --problem "Influence Maximization" --train_dataset HK --test_dataset DBLP
# python test.py --problem "Influence Maximization" --train_dataset HK --test_dataset YouTube
# python test.py --problem "Influence Maximization" --train_dataset HK --test_dataset Skitter


# python finetune.py --problem "Maximum Coverage" --dataset Facebook
# python finetune.py --problem "Maximum Coverage" --dataset Wiki
# python finetune.py --problem "Maximum Coverage" --dataset Deezer
# python finetune.py --problem "Maximum Coverage" --dataset Slashdot
# python finetune.py --problem "Maximum Coverage" --dataset Twitter
# python finetune.py --problem "Maximum Coverage" --dataset DBLP
# python finetune.py --problem "Maximum Coverage" --dataset YouTube
# python finetune.py --problem "Maximum Coverage" --dataset Skitter

# python test.py --problem "Maximum Coverage" --train_dataset Facebook --test_dataset Facebook --k 1000
# python test.py --problem "Maximum Coverage" --train_dataset Wiki --test_dataset Wiki --k 1000
# python test.py --problem "Maximum Coverage" --train_dataset Deezer --test_dataset Deezer --k 1000
# python test.py --problem "Maximum Coverage" --train_dataset Slashdot --test_dataset Slashdot --k 1000
# python test.py --problem "Maximum Coverage" --train_dataset Twitter --test_dataset Twitter --k 5000
# python test.py --problem "Maximum Coverage" --train_dataset DBLP --test_dataset DBLP    --k 5000
# python test.py --problem "Maximum Coverage" --train_dataset YouTube --test_dataset YouTube --k 5000
# python test.py --problem "Maximum Coverage" --train_dataset Skitter --test_dataset Skitter --k 5000


# Training
# python finetune.py --problem "Maximum Coverage" --dataset Facebook &
# python finetune.py --problem "Maximum Coverage" --dataset Wiki &
# python finetune.py --problem "Maximum Coverage" --dataset Deezer &
# python finetune.py --problem "Maximum Coverage" --dataset Slashdot &
# python finetune.py --problem "Maximum Coverage" --dataset Twitter &
# python finetune.py --problem "Maximum Coverage" --dataset DBLP &
# python finetune.py --problem "Maximum Coverage" --dataset YouTube &
# python finetune.py --problem "Maximum Coverage" --dataset Skitter &
# wait  # waits until all background jobs finish

# # Testing
# python test.py --problem "Maximum Coverage" --train_dataset Facebook --test_dataset Facebook --k 1000 &
# python test.py --problem "Maximum Coverage" --train_dataset Wiki --test_dataset Wiki --k 1000 &
# python test.py --problem "Maximum Coverage" --train_dataset Deezer --test_dataset Deezer --k 1000 &
# python test.py --problem "Maximum Coverage" --train_dataset Slashdot --test_dataset Slashdot --k 1000  &
# python test.py --problem "Maximum Coverage" --train_dataset Twitter --test_dataset Twitter --k 5000 &
# python test.py --problem "Maximum Coverage" --train_dataset DBLP --test_dataset DBLP --k 5000 &
# python test.py --problem "Maximum Coverage" --train_dataset YouTube --test_dataset YouTube --k 5000 &
# python test.py --problem "Maximum Coverage" --train_dataset Skitter --test_dataset Skitter --k 5000 &
# wait


# python finetune.py --problem "Maximum Cut" --dataset Facebook
# python finetune.py --problem "Maximum Cut" --dataset Wiki
# python finetune.py --problem "Maximum Cut" --dataset Deezer
# python finetune.py --problem "Maximum Cut" --dataset Slashdot
# python finetune.py --problem "Maximum Cut" --dataset Twitter
# python finetune.py --problem "Maximum Cut" --dataset DBLP
# python finetune.py --problem "Maximum Cut" --dataset YouTube
# python finetune.py --problem "Maximum Cut" --dataset Skitter
# wait

# python test.py --problem "Maximum Cut" --train_dataset Facebook --test_dataset Facebook --k 1000 &
# python test.py --problem "Maximum Cut" --train_dataset Wiki --test_dataset Wiki --k 1000 &
# python test.py --problem "Maximum Cut" --train_dataset Deezer --test_dataset Deezer --k 1000 &
# python test.py --problem "Maximum Cut" --train_dataset Slashdot --test_dataset Slashdot --k 1000 &
# python test.py --problem "Maximum Cut" --train_dataset Twitter --test_dataset Twitter --k 5000 &
# python test.py --problem "Maximum Cut" --train_dataset DBLP --test_dataset DBLP    --k 5000 &
# python test.py --problem "Maximum Cut" --train_dataset YouTube --test_dataset YouTube --k 5000 &
# python test.py --problem "Maximum Cut" --train_dataset Skitter --test_dataset Skitter --k 5000 &
# wait


python finetune.py --problem "Influence Maximization" --dataset Facebook
python finetune.py --problem "Influence Maximization" --dataset Wiki
python finetune.py --problem "Influence Maximization" --dataset Deezer
python finetune.py --problem "Influence Maximization" --dataset Slashdot
python finetune.py --problem "Influence Maximization" --dataset Twitter
python finetune.py --problem "Influence Maximization" --dataset DBLP
python finetune.py --problem "Influence Maximization" --dataset YouTube
python finetune.py --problem "Influence Maximization" --dataset Skitter

python test.py --problem "Influence Maximization" --train_dataset Facebook --test_dataset Facebook --k 1000
python test.py --problem "Influence Maximization" --train_dataset Wiki --test_dataset Wiki --k 1000
python test.py --problem "Influence Maximization" --train_dataset Deezer --test_dataset Deezer --k 1000
python test.py --problem "Influence Maximization" --train_dataset Slashdot --test_dataset Slashdot --k 1000
python test.py --problem "Influence Maximization" --train_dataset Twitter --test_dataset Twitter --k 5000
python test.py --problem "Influence Maximization" --train_dataset DBLP --test_dataset DBLP    --k 5000
# python test.py --problem "Influence Maximization" --train_dataset YouTube --test_dataset YouTube --k 5000
# python test.py --problem "Influence Maximization" --train_dataset Skitter --test_dataset Skitter --k 5000




# python finetune.py --problem "Maximum Cut Weighted" --dataset Facebook
# python finetune.py --problem "Maximum Cut Weighted" --dataset Wiki
# python finetune.py --problem "Maximum Cut Weighted" --dataset Deezer
# python finetune.py --problem "Maximum Cut Weighted" --dataset Slashdot
# python finetune.py --problem "Maximum Cut Weighted" --dataset Twitter
# python finetune.py --problem "Maximum Cut Weighted" --dataset DBLP
# python finetune.py --problem "Maximum Cut Weighted" --dataset YouTube
# python finetune.py --problem "Maximum Cut Weighted" --dataset Skitter

# python test.py --problem "Maximum Coverage Weighted" --train_dataset Facebook --test_dataset Facebook --k 1000 &
# python test.py --problem "Maximum Coverage Weighted" --train_dataset Wiki --test_dataset Wiki --k 1000 &
# python test.py --problem "Maximum Coverage Weighted" --train_dataset Deezer --test_dataset Deezer --k 1000 &
# python test.py --problem "Maximum Coverage Weighted" --train_dataset Slashdot --test_dataset Slashdot --k 1000 &
# python test.py --problem "Maximum Coverage Weighted" --train_dataset Twitter --test_dataset Twitter --k 5000 &
# python test.py --problem "Maximum Coverage Weighted" --train_dataset DBLP --test_dataset DBLP --k 5000 &
# python test.py --problem "Maximum Coverage Weighted" --train_dataset YouTube --test_dataset YouTube --k 5000 &
# python test.py --problem "Maximum Coverage Weighted" --train_dataset Skitter --test_dataset Skitter --k 5000 &

# wait

# python test.py --problem "Maximum Cut Weighted" --train_dataset Facebook --test_dataset Facebook --k 1000 &
# python test.py --problem "Maximum Cut Weighted" --train_dataset Wiki --test_dataset Wiki --k 1000 &
# python test.py --problem "Maximum Cut Weighted" --train_dataset Deezer --test_dataset Deezer --k 1000 &
# python test.py --problem "Maximum Cut Weighted" --train_dataset Slashdot --test_dataset Slashdot --k 1000 &
# python test.py --problem "Maximum Cut Weighted" --train_dataset Twitter --test_dataset Twitter --k 5000 &
# python test.py --problem "Maximum Cut Weighted" --train_dataset DBLP --test_dataset DBLP    --k 5000 &
# python test.py --problem "Maximum Cut Weighted" --train_dataset YouTube --test_dataset YouTube --k 5000 &
# python test.py --problem "Maximum Cut Weighted" --train_dataset Skitter --test_dataset Skitter --k 5000 &


# python finetune.py --problem "Influence Maximization Weighted" --dataset Facebook
# python finetune.py --problem "Influence Maximization Weighted" --dataset Wiki
# python finetune.py --problem "Influence Maximization Weighted" --dataset Deezer
# python finetune.py --problem "Influence Maximization Weighted" --dataset Slashdot
# python finetune.py --problem "Influence Maximization Weighted" --dataset Twitter
# python finetune.py --problem "Influence Maximization Weighted" --dataset DBLP
# python finetune.py --problem "Influence Maximization Weighted" --dataset YouTube
# python finetune.py --problem "Influence Maximization Weighted" --dataset Skitter

# python test.py --problem "Influence Maximization Weighted" --train_dataset Facebook --test_dataset Facebook --k 1000
# python test.py --problem "Influence Maximization Weighted" --train_dataset Wiki --test_dataset Wiki --k 1000
# python test.py --problem "Influence Maximization Weighted" --train_dataset Deezer --test_dataset Deezer --k 1000
# python test.py --problem "Influence Maximization Weighted" --train_dataset Slashdot --test_dataset Slashdot --k 1000
# python test.py --problem "Influence Maximization Weighted" --train_dataset Twitter --test_dataset Twitter --k 5000
# python test.py --problem "Influence Maximization Weighted" --train_dataset DBLP --test_dataset DBLP    --k 5000
# python test.py --problem "Influence Maximization Weighted" --train_dataset YouTube --test_dataset YouTube --k 5000
# python test.py --problem "Influence Maximization Weighted" --train_dataset Skitter --test_dataset Skitter --k 5000