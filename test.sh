Q1=query_shot1
Q2=query_shot2
Q3=query_shot3

for FOLD in 1
do
    python3 test.py --query_shot=$Q1 --fold=$FOLD
done

# for FOLD in 1 2 3 4 5
# do
#     python3 test.py --query_shot=$Q2 --fold=$FOLD
# done

# for FOLD in 1 2 3 4 5
# do
#     python3 test.py --query_shot=$Q3 --fold=$FOLD
# done

