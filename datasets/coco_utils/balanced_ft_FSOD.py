import itertools
import random
import os
import xml.etree.ElementTree as ET
from fvcore.common.file_io import PathManager

from detectron2.utils.store_non_list import Store

VOC_CLASS_NAMES_COCOFIED = [
    "airplane",  "dining table", "motorcycle",
    "potted plant", "couch", "tv"
]

BASE_VOC_CLASS_NAMES = [
    "aeroplane", "diningtable", "motorbike",
    "pottedplant",  "sofa", "tvmonitor"
]

VOC_CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

T2_CLASS_NAMES = [
        'lipstick', 'sandal', 'crocodile', 'football helmet', 'umbrella', 'hartebeest', 'houseplant', 'modem', 'antelope', 
        'woodpecker', 'palm tree', 'shower cap', 'mask', 'box', 'handkerchief', 'falafel', 'swan', 'miniskirt', 'monkey', 
        'cookie', 'scissors', 'clipper', 'snowboard', 'croquette', 'house finch', 'hedgehog', 'penguin', 'barrel', 'butterfly fish', 
        'lesser scaup', 'wall clock', 'barbell', 'hair slide', 'strawberry', 'window blind', 'butterfly', 'television', 'arabian camel', 
        'cake', 'pill bottle', 'springbok', 'camper', 'punching bag', 'picture frame', 'face powder', 'jaguar', 'basketball player', 
        'tomato', 'isopod', 'balloon', 'bumper car', 'wisent', 'hip', 'vase', 'shirt', 'waffle', 'carrot', 'candle', 'wicket', 
        'flute', 'medicine ball', 'bagel', 'orange', 'wheelchair', 'golf ball', 'unicycle', 'surfboard', 'cattle', 'parachute', 
        'sweet orange', 'snowshoe', 'candy', 'turkey', 'pillow', 'column', 'king charles spaniel', 'jacket', 'crane', 'scoter', 
        'dumbbell', 'dagger', 'wine glass', 'guitar', 'slide rule', 'steel drum', 'sports car', 'go kart', 'shrimp', 'worm', 
        'hamburger', 'cucumber', 'gearing', 'radish', 'tostada', 'french loaf', 'granny smith', 'sorrel', 'ibex', 'rain barrel', 
        'quail', 'alpaca', 'rhodesian ridgeback', 'mongoose', 'red backed sandpiper', 'penlight', 'bicycle wheel', 'shelf', 'pancake', 
        'samoyed', 'helicopter', 'pay phone', 'barber chair', 'perfume', 'wool', 'sword', 'ballplayer', 'ipod', 'goose', 'pretzel', 
        'coin', 'broccoli', 'malamute', 'reel', 'mountain goat', 'mule', 'tusker', 'cabbage', 'longwool', 'sheep', 'apple', 'flag', 
        'shopping cart', 'marble', 'horse', 'duck', 'salad', 'lemon', 'handgun', 'shuttlecock', 'red breasted merganser', 'shutter', 
        'stamp', 'backpack', 'printer', 'mug', 'snowmobile', 'boot', 'bowl', 'book', 'tin can', 'football', 'human leg', 'letter opener', 
        'countertop', 'elephant', 'ladybug', 'curtain', 'wine', 'canopic jar', 'warthog', 'van', 'oil filter', 'envelope', 'pen', 
        'petri dish', 'bubble', 'doll', 'bus', 'african crocodile', 'bikini', 'brambling', 'siamang', 'bison', 'snorkel', 'flying disc', 
        'microwave oven', 'stethoscope', 'loafer', 'burrito', 'kite balloon', 'wallet', 'mushroom', 'laundry cart', 'teddy bear', 'nail', 
        'sausage dog', 'bottle', 'raccoon', 'rifle', 'peach', 'king penguin', 'laptop', 'diver', 'centipede', 'rake', 'tiger', 'watch', 
        'drake', 'bald eagle', 'cat', 'ladder', 'sparrow', 'retriever', 'coffee table', 'plastic bag', 'slot', 'brown bear', 'frog', 
        'jeans', 'switchblade', 'harp', 'orangutan', 'accordion', 'chacma', 'guenon', 'pig', 'porcupine', 'dolphin', 'car wheel', 'owl', 
        'dandie dinmont', 'flowerpot', 'guanaco', 'motorcycle', 'corn', 'hen', 'calculator', 'african hunting dog', 'tap', 'kangaroo', 
        'pajama', 'lavender', 'hay', 'dingo', 'tennis ball', 'meat loaf', 'kid', 'jellyfish', 'whistle', 'tank car', 'dungeness crab', 
        'bust', 'dice', 'pop bottle', 'wok', 'roller skates', 'oar', 'yellow lady\'s slipper', 'mango', 'mountain sheep', 'bread', 'zebu',
        'crossword puzzle', 'computer monitor', 'daisy', 'kimono', 'sombrero', 'basenji', 'desk', 'solar dish', 'cheetah', 'bell', 
        'ice cream', 'gazelle', 'agaric', 'tart', 'doughnut', 'meatball', 'grapefruit', 'patas', 'paddle', 'swing', 'dutch oven', 
        'pear', 'military uniform', 'vestment', 'kite', 'eagle', 'towel', 'cavy', 'coffee', 'mustang', 'standard poodle', 
        'chesapeake bay retriever', 'coffee mug', 'deer', 'gorilla', 'bearskin', 'whale', 'cello', 'lion', 'taxi', 'safety pin', 
        'sulphur crested cockatoo', 'flamingo', 'shark', 'eider', 'picket fence', 'human arm', 'trumpet'
]

T3_CLASS_NAMES = [
        'french fries','dhole','spaghetti squash','syringe','african elephant','lobster','rose','human hand','coral fungus',
        'pelican','anchovy pear','oystercatcher','lamp','gyromitra','bat','african grey','ostrich','knee pad','trombone','swim cap',
        'human beard','hot dog','chicken','hatchet','leopard','elk','alarm clock','drum','taco','digital clock','squash racket',
        'starfish','train','belt','mallet','refrigerator','greyhound','ram','dog bed','racer','morel','bell pepper','drumstick',
        'loveseat','bovine','bullet train','bernese mountain dog','motor scooter','vervet','quince','blenheim spaniel','infant bed',
        'snipe','training bench','milk','mixing bowl','marmoset','knife','cutting board','ring binder','studio couch','filing cabinet',
        'bee','caterpillar','sofa bed','dodo','cowboy boot','violin','buckeye','prairie chicken','traffic light','airplane',
        'siberian husky','ballpoint','mountain tent','jockey','border collie','ice skate','closet','button','stuffed tomato',
        'lovebird','canary','jinrikisha','toilet paper','canoe','pony','killer whale','spoon','indian elephant','fox','tennis racket',
        'acorn squash','macaw','bolete','fiddler crab','mobile home','dressing table','chimpanzee','jack o\' lantern','red panda',
        'toast','cannon','nipple','entlebucher','stool','groom','sarong','cauliflower','apiary','english foxhound','deck chair',
        'car door','zucchini','rugby ball','labrador retriever','wallaby','polar bear','acorn','bench','pizza','short pants',
        'standard schnauzer','lampshade','fork','hog','barge','male horse','bow and arrow','martin','kettle','goldfish','mirror',
        'loudspeaker','snail','poster','drill','tie','plum','bale','partridge','gondola','water jug','scale','shoji','shield',
        'american lobster','falcon','bull','nailfile','poodle','jackfruit','heifer','whippet','remote control','mitten','eggnog',
        'horn','hamster','weimaraner','volleyball','twin bed','english springer','stationary bicycle','dishwasher','dowitcher',
        'rhesus','limousine','norwich terrier','sail','shorts','toothbrush','custard apple','wassail','bib','bookcase','baseball glove',
        'computer mouse','bullet','bartlett','otter','computer keyboard','shower','brace','pick','teapot','carthorse','human foot',
        'parking meter','ruminant','clog','screw','burro','mountain bike','ski','beaker','sunscreen','packet','castle','mobile phone',
        'madagascar cat','suitcase','radio telescope','sock','cupboard','wild sheep','crab','stuffed peppers','okapi','common fig',
        'missile','swimwear','saucer','popcorn','coat','bighorn','plate','grizzly','stairs','pineapple','parrot','fountain','jar',
        'binoculars','rambutan','tent','pencil case','mortarboard','raspberry','gar','andiron','paintbrush','running shoe','turnstile',
        'mouse','leonberg','red wine','sewing machine','open face sandwich','magpie','metal screw','west highland white terrier',
        'handbag','saxophone','panda','flashlight','boxer','lorikeet','interceptor','baseball bat','ruddy turnstone','colobus',
        'golf cart','pan','white stork','banana','billiard table','stinkhorn','american coot','tower','trailer truck',
        'washing machine','bride','afghan hound','motorboat','lizard','bassoon','brassiere','quesadilla','goblet','llama',
        'folding chair','spoonbill','ant','workhorse','crown','pimento','anemone fish','oven','sea lion','ewe','pitcher',
        'megalith','pool ball','chest of drawers','macaque','kit fox','oryx','sleeve','crutch','plug','battery','black stork',
        'hippopotamus','saluki','bath towel','artichoke','seat belt','bee eater','baboon'
]

T4_CLASS_NAMES = [
        'dairy cattle','sleeping bag','panpipe','gemsbok','microphone','lynx','camel','rabbit','rocket','toilet','albatross',
        'spider','comb','snow goose','cetacean','camera','bucket','packhorse','palm','vending machine','butternut squash','loupe',
        'ox','celandine','appenzeller','pomegranate','bathtub','vulture','crampon','jug','backboard','european gallinule','goat',
        'cowboy hat','parsnip','jersey','slide','guava','wrench','cardoon','scuba diver','broom','giant schnauzer','stretcher',
        'balance beam','gordon setter','necklace','scoreboard','horizontal bar','stop sign','sushi','staffordshire bullterrier',
        'gas stove','conch','cherry','jam','salmon','matchstick','tank','armadillo','black swan','sailboat','assault rifle','snake',
        'thatch','tripod','hook','wild boar','cocktail','ski pole','zebra','toaster','armchair','lab coat','goldfinch','guinea pig',
        'frying pan','pinwheel','water buffalo','pasta','chain','ocarina','impala','swallow','mailbox','langur','cock','hyena',
        'marimba','hound','truck','blue jay','knot','saw','eskimo dog','pembroke','sink','lighthouse','skateboard','sealyham terrier',
        'cricket ball','dragonfly','snowplow','italian greyhound','shih tzu','scotch terrier','yawl','screwdriver','organ','lighter',
        'giraffe','submarine','dung beetle','scorpion','dugong','academic gown','blanket','honeycomb','timber wolf','cream','minibus',
        'joystick','cart','koala','speedboat','guacamole','flagpole','raven','drawer','honey','diaper','chessman','fire hydrant',
        'club sandwich','potato','porch','banjo','gown','crate','peg','hammer','aquarium','whooping crane','headboard','okra',
        'trench coat','avocado','cayuse','paper towel','large yellow lady\'s slipper','ski mask','dough','bassarisk','bridal gown',
        'wardrobe','terrapin','yacht','soap dispenser','saddle','redbone','asparagus','skunk','chainsaw','shower curtain','jennet',
        'spatula','ambulance','school bus','otterhound','irish terrier','carton','abaya','submarine sandwich','axe','ruler',
        'window shade','measuring cup','wooden spoon','yurt','scarf','squirrel','flat coated retriever','bull mastiff','cardigan',
        'river boat','irish wolfhound','oxygen mask','propeller','earthstar','black footed ferret','tea','rocking chair','whisk',
        'food processor','beach wagon','litchi','pigeon','tick','stapler','oboe'
]

UNK_CLASS = ["unknown"]

# Change this accodingly for each task t*
known_classes = list(itertools.chain(VOC_CLASS_NAMES, T2_CLASS_NAMES))
train_files = ['../VOC2007/ImageSets/Main/t2_train_fsod.txt','../VOC2007/ImageSets/Main/t1_train.txt']

# known_classes = list(itertools.chain(VOC_CLASS_NAMES))
# train_files = ['/home/fk1/workspace/OWOD/datasets/VOC2007/ImageSets/Main/train.txt']
annotation_location = '/home/causal_ws/OWOD/datasets/VOC2007/Annotations'

items_per_class = 1
dest_file = './ImageSets/Main/t2_ft_fsod_' + str(items_per_class) + '.txt'

file_names = []
for tf in train_files:
    with open(tf, mode="r") as myFile:
        file_names.extend(myFile.readlines())

random.shuffle(file_names)

image_store = Store(len(known_classes), items_per_class)

current_min_item_count = 0

for fileid in file_names:
    fileid = fileid.strip()
    anno_file = os.path.join(annotation_location, fileid + ".xml")

    with PathManager.open(anno_file) as f:
        tree = ET.parse(f)

    for obj in tree.findall("object"):
        cls = obj.find("name").text
        if cls in VOC_CLASS_NAMES_COCOFIED:
            cls = BASE_VOC_CLASS_NAMES[VOC_CLASS_NAMES_COCOFIED.index(cls)]
        if cls in known_classes:
            image_store.add((fileid,), (known_classes.index(cls),))

    current_min_item_count = min([len(items) for items in image_store.retrieve(-1)])
    print(current_min_item_count)
    if current_min_item_count == items_per_class:
        break

filtered_file_names = []
for items in image_store.retrieve(-1):
    filtered_file_names.extend(items)

print(image_store)
print(len(filtered_file_names))
print(len(set(filtered_file_names)))

filtered_file_names = set(filtered_file_names)
filtered_file_names = map(lambda x: x + '\n', filtered_file_names)

with open(dest_file, mode="w") as myFile:
    myFile.writelines(filtered_file_names)

print('Saved to file: ' + dest_file)
