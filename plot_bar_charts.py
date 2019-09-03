import matplotlib.pyplot as plt



plt.rcParams.update({'font.size': 14})


### other patterns: https://www.w3resource.com/graphics/matplotlib/barchart/matplotlib-barchart-exercise-17.php
# Input data; groupwise
seq2seq = [0.86, 0.89, 0.88, 0.90, 0.87]
lstm =    [0.84, 0.87, 0.85, 0.865, 0.85]
ngram =   [0.82, 0.83, 0.83, 0.845, 0.84]
cnn = [0.78, 0.73, 0.73, 0.72, 0.70]
patterns = [ "|" , "\\" , "/" , "+" , "-", ".", "*","x", "o", "O" ]

labels = ['Houston', 'San Antonio', 'Austin', 'Dallas', 'Corpus Christi']

# Setting the positions and width for the bars
pos = list(range(len(seq2seq)))
width = 0.2  # the width of a bar

# Plotting the bars
fig, ax = plt.subplots(figsize=(10, 6))

bar1 = plt.bar(pos, seq2seq, width,
               alpha=0.5,
               color='w',
               hatch=patterns[7],  # this one defines the fill pattern
               edgecolor='black',
               label=labels[0])

plt.bar([p + width for p in pos], lstm, width,
        alpha=0.5,
        color='w',
        hatch=patterns[6],
        edgecolor='black',
        label=labels[1])


plt.bar([p + width * 2 for p in pos], ngram, width,
        alpha=0.5,
        color='w',
        hatch=patterns[8],
        edgecolor='black',
        label=labels[3])

plt.bar([p + width * 3 for p in pos], cnn, width,
        alpha=0.5,
        color='w',
        hatch=patterns[1],
        edgecolor='black',
        label=labels[3])

# Setting axis labels and ticks
ax.set_ylabel('Accuracy')
# ax.set_xlabel('City')
ax.set_title('Need prediction accuracy by city')
ax.set_xticks([p + 1 * width for p in pos])
ax.set_xticklabels(labels)

# Setting the x-axis and y-axis limits
plt.xlim(min(pos) - width, max(pos) + width * 5)
plt.ylim([0.6, 1])

# Adding the legend and showing the plot
plt.legend(['Seq2seq', 'LSTM', 'N-gram', 'CNN'], loc='upper right')
# plt.grid()
plt.show()