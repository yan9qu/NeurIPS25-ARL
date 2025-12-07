benchmarks = {
    'MintRec':{
        'intent_labels': [
                    'Complain', 'Praise', 'Apologise', 'Thank', 'Criticize', 
                    'Agree', 'Taunt', 'Flaunt', 
                    'Joke', 'Oppose', 
                    'Comfort', 'Care', 'Inform', 'Advise', 'Arrange', 'Introduce', 'Leave', 
                    'Prevent', 'Greet', 'Ask for help' 
        ],
        'binary_maps': {
                    'Complain': 'Emotion', 'Praise':'Emotion', 'Apologise': 'Emotion', 'Thank':'Emotion', 'Criticize': 'Emotion',
                    'Care': 'Emotion', 'Agree': 'Emotion', 'Taunt': 'Emotion', 'Flaunt': 'Emotion',
                    'Joke':'Emotion', 'Oppose': 'Emotion', 
                    'Inform':'Goal', 'Advise':'Goal', 'Arrange': 'Goal', 'Introduce': 'Goal', 'Leave':'Goal',
                    'Prevent':'Goal', 'Greet': 'Goal', 'Ask for help': 'Goal', 'Comfort': 'Goal'
        },
        'binary_intent_labels': ['Emotion', 'Goal'],
        'max_seq_lengths':{
            'text': 30, # truth: 26 
            'video': 230, # truth: 225
            'audio': 480, # truth: 477
        },
        'feat_dims':{
            'text': 768,
            'video': 256,
            'audio': 768
        }
    },

    'UR_FUNNY':{
        'binary_intent_labels': [0, 1],
        'max_seq_lengths':{
            'text': 130, # truth: 26 
            'video': 130, # truth: 225
            'audio': 130, # truth: 477
        },
        'feat_dims':{
            'text': 768,
            'video': 371,
            'audio': 81
        }
    },

    'MOSI':{
        'intent_labels':[0,1,2,3,4,5,6],
        'label_len':3,
        'max_seq_lengths':{
            'text': 200, 
            'video': 50, 
            'audio': 50, 
        },
        'feat_dims':{
            'text': 768,
            'video': 20,
            'audio': 5
        }
    },
    'MOSEI':{
        'intent_labels':[0,1,2,3,4,5,6],
        'label_len':3,
        'max_seq_lengths':{
            'text': 320, 
            'video': 50, 
            'audio': 50, 
        },
        'feat_dims':{
            'text': 768,
            'video': 35,
            'audio': 74
        }
    }
}
