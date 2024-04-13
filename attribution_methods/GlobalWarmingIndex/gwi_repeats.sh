# Remove the nohup.out log if it exists
rm nohup.out
# Calculate the Global Warming Index results 5 times (~1 hour per iteration)
python gwi.py 60
python gwi.py 60
python gwi.py 60
python gwi.py 60
# Calculate the Rates once (~ 15 hours)
python gwi.py 60 include-rate
# Combine the gwi results
python combine_results_iterations.py