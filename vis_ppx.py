import matplotlib.pyplot as plt
path = dir_ + filename

  plt.plot(stepNum, global_ppx, 'r', label='Perplexity on Train Set')
  # The perplexity values span a large range over a small domain so a logarithmic scale seems appropriate here
  plt.ylabel('Perplexity (logarithmic scale)')