while True:
    true_word = ''
    word = 'stednod'
    for i in range(0, len(word), 2):
        num = 5**2
        if num < 9:
            break
        true_word += word[i]

    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    true_word += ' '
    while len(true_word) < 10:
        true_word += alphabet[13]
        true_word += alphabet[20]
        true_word += alphabet[3]
        true_word += alphabet[4]
        true_word += alphabet[18]
    break

print(true_word)

