#!/bin/bash
# Quick diagnostic script to check backward elimination job status

LOG_DIR=/user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/one_ff/logs/back_elim

echo "=== Backward Elimination Job Status Check ==="
echo ""

# Check SLURM queue for running/pending jobs
echo "1. Current SLURM jobs:"
squeue -u $USER --name=one_ff_back_elim --format="%.10i %.9P %.20j %.8u %.2t %.10M %.6D %R" 2>/dev/null || echo "   (No jobs in queue or squeue not available)"
echo ""

# Check recent log files
echo "2. Recent log files (last 10):"
if [ -d "$LOG_DIR" ]; then
    ls -lt "$LOG_DIR"/*.err 2>/dev/null | head -10 || echo "   (No error logs found)"
    echo ""
    
    echo "3. Units that completed successfully:"
    grep -l "completed successfully" "$LOG_DIR"/*.err 2>/dev/null | wc -l || echo "   0"
    echo ""
    
    echo "4. Units stuck after 'Set up logging configuration':"
    for f in "$LOG_DIR"/*.err; do
        if [ -f "$f" ]; then
            if grep -q "Set up logging configuration" "$f" && ! grep -q "\[PYTHON\]\[DEBUG\]" "$f"; then
                basename "$f"
            fi
        fi
    done | head -20
    echo ""
    
    echo "5. Sample of last debug message from recent logs:"
    for f in $(ls -t "$LOG_DIR"/*.err 2>/dev/null | head -5); do
        if [ -f "$f" ]; then
            echo "  $(basename $f): $(tail -1 $f)"
        fi
    done
else
    echo "   Log directory not found: $LOG_DIR"
fi

echo ""
echo "=== To investigate specific unit ==="
echo "tail -50 $LOG_DIR/one_ff_back_elim_JOBID_UNITIDX.err"
