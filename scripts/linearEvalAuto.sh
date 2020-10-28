for i in {1..27}
do
    echo "Evaluating REINFORCE/batch0 $i"
    python -m trlfpi.eval.linearSystemEvaluation REINFORCE/batch0 $i --plots
done
