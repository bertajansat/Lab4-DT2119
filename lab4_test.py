import torch
from lab4_proto import dataProcessing, test_audio_transform, levenshteinDistance
from lab4_main import wer, cer
example = torch.load('lab4_example.pt')

# 3.2: 
spectrograms, labels, input_lengths, label_lengths = dataProcessing(example['data'], test_audio_transform)


print(torch.allclose(spectrograms, example['spectrograms'], atol=1e-4))
print(torch.equal(labels, example['labels']))
print(torch.equal(torch.tensor(input_lengths), torch.tensor(example['input_lengths'])))
print(torch.equal(torch.tensor(label_lengths), torch.tensor(example['label_lengths'])))

str_1 = "Hello new world"
str2 = "Helo big world"
ex_cer = cer(str_1,str2)
ex_wer = wer(str_1,str2)
print(f"CER={ex_cer}, WER = {ex_wer}")

# Some utterances are:
#New utterance!
#I HEARD MY OWN VOICE REJOINED MONTONI STERNLY AND NOTHING ELSE

#New utterance!
#EVEN NOW THE RUINS MAY BE IMPASSABLE GRAHAM REGARDED HIM DOUBTFULLY AND FOLLOWED HIM THEY WENT UP THE STEPPED PLATFORMS TO THE SWIFTEST ONE AND THERE ASANO ACCOSTED A LABOURER THE ANSWERS TO HIS QUESTIONS WERE IN THE THICK VULGAR SPEECH WHAT DID HE SAY ASKED GRAHAM

#New utterance!
#UNSHADED WITH NO CHARM OF PAST ASSOCIATION ONLY A MEMORY OF FORCED HUMAN TOIL NOW THEN AND BEFORE THE WAR THEY ARE NOT HAPPY THESE BLACK MEN WHOM WE MEET THROUGHOUT THIS REGION

#New utterance!
#THESE MELANCHOLY OPENINGS WHICH TAKE PLACE IN THE GLOOM BEFORE DESPAIR ARE TEMPTING MARIUS THRUST ASIDE THE BAR WHICH HAD SO OFTEN ALLOWED HIM TO PASS EMERGED FROM THE GARDEN AND SAID I WILL GO

#New utterance!
#FOR SHE WAS SHUT UP IN HER TOWER WEEPING BITTERLY THE WICKED BROTHERS STAMPED AND FOAMED WITH RAGE WHEN THEY SAW THE FAILURE OF THEIR WICKED DESIGNS BUT THE KING WAS OVERCOME BY A SUDDEN TERROR

#New utterance!
#I AM GOING TO SEE MISS GOWER AUNT SHE VENTURED TO SAY ONE MORNING AT THE BREAKFAST TABLE
