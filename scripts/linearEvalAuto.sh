for i in {0..24}
do
    echo "Evaluating ACD $i"
    python -m trlfpi.eval.evalACD ACD $i --plots
done
