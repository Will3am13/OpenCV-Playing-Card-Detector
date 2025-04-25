import cv2
import pytesseract
import re
import os
import shutil
import numpy as np

# --- Configuration ---
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
IMAGE_PATH = r"C:\Users\willd\Pictures\turn_raise.jpg"
OUTPUT_DIR = 'roi_output'
CARD_TRAIN_DIR = 'Card_Imgs'

# --- Clear and recreate output directory ---
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)
print(f"Created directory: {OUTPUT_DIR}")

# --- ROI Definitions ---
PLAYER_ROIS = {
    "Hero": {  # Bottom center (USER)
        "name": (941, 870, 126, 43),
        "stack": (947, 916, 126, 43),
        "cards": [(915, 782, 38, 81), (955, 780, 38, 81)],
    },
    "Player_Seven_Clock": {
        "name": (448, 781, 126, 43),
        "stack": (438, 817, 126, 43),
        "bet": (648, 696, 84, 34),
        "status": (457, 860, 91, 22)
    },
    "Player_Nine_Clock": {
        "name": (330, 533, 126, 43),
        "stack": (332, 574, 126, 43),
        "bet": (564, 551, 84, 34),
        "status": (347, 610, 91, 22)
    },
    "Player_Ten_Clock": {
        "name": (410, 337, 126, 43),
        "stack": (408, 337, 126, 43),
        "bet": (638, 430, 89, 34),
        "status": (427, 416, 91, 22)
    },
    "Player_Noon_Clock": {
        "name": (862, 250, 126, 43),
        "stack": (859, 292, 126, 43),
        "bet": (914, 425, 84, 34),
        "status": (877, 330, 91, 22)
    },
    "Player_One_Clock": {
        "name": (1394, 339, 126, 43),
        "stack": (1391, 380, 126, 43),
        "bet": (1184, 430, 84, 34),
        "status": (1405, 420, 91, 22)
    },
    "Player_Three_Clock": {
        "name": (1476, 531, 126, 43),
        "stack": (1474, 575, 126, 43),
        "bet": (1275, 600, 84, 34),
        "status": (1489, 614, 91, 22)
    },
    "Player_Four_Clock": {
        "name": (1363, 778, 126, 50),
        "stack": (1362, 825, 126, 50),
        "bet": (1200, 700, 84, 34),
        "status": (1382, 861, 91, 22)
    },
}

COMMUNITY_CARD_ROIS = [
    (746, 528, 34, 77),  # Flop 1
    (833, 528, 34, 77),  # Flop 2
    (920, 528, 34, 77),  # Flop 3
    (1007, 528, 34, 77),  # Turn
    (1094, 528, 34, 77),  # River
]

POT_ROI = (878, 488, 170, 35)  # "POT: 29,600"

# --- Card Detection Constants ---
# Adaptive threshold levels
BKG_THRESH = 60
CARD_THRESH = 30

# Width and height of card corner, where rank and suit are
CORNER_WIDTH = 32
CORNER_HEIGHT = 84

# Dimensions of rank train images
RANK_WIDTH = 70
RANK_HEIGHT = 125

# Dimensions of suit train images
SUIT_WIDTH = 70
SUIT_HEIGHT = 100

RANK_DIFF_MAX = 2000
SUIT_DIFF_MAX = 700


# --- Card Train Classes ---
class Train_ranks:
    """Structure to store information about train rank images."""

    def __init__(self):
        self.img = []  # Thresholded, sized rank image loaded from hard drive
        self.name = "Placeholder"


class Train_suits:
    """Structure to store information about train suit images."""

    def __init__(self):
        self.img = []  # Thresholded, sized suit image loaded from hard drive
        self.name = "Placeholder"


class Query_card:
    """Structure to store information about query cards in the camera image."""

    def __init__(self):
        self.contour = []  # Contour of card
        self.width, self.height = 0, 0  # Width and height of card
        self.corner_pts = []  # Corner points of card
        self.center = []  # Center point of card
        self.warp = []  # 200x300, flattened, grayed image
        self.rank_img = []  # Thresholded, sized image of card's rank
        self.suit_img = []  # Thresholded, sized image of card's suit
        self.best_rank_match = "Unknown"  # Best matched rank
        self.best_suit_match = "Unknown"  # Best matched suit
        self.rank_diff = 0  # Difference between rank image and best matched train rank image
        self.suit_diff = 0  # Difference between suit image and best matched train suit image


# --- Card Detection Functions ---
def load_ranks(filepath):
    """Loads rank images from directory specified by filepath. Stores
    them in a list of Train_ranks objects."""
    train_ranks = []
    i = 0

    for Rank in ['Ace', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven',
                 'Eight', 'Nine', 'Ten', 'Jack', 'Queen', 'King']:
        train_ranks.append(Train_ranks())
        train_ranks[i].name = Rank
        filename = Rank + '.jpg'
        filepath_full = os.path.join(filepath, filename)
        train_ranks[i].img = cv2.imread(filepath_full, cv2.IMREAD_GRAYSCALE)
        if train_ranks[i].img is None:
            print(f"Warning: Could not load {filepath_full}")
        i = i + 1

    return train_ranks


def load_suits(filepath):
    """Loads suit images from directory specified by filepath. Stores
    them in a list of Train_suits objects."""
    train_suits = []
    i = 0

    for Suit in ['Spades', 'Diamonds', 'Clubs', 'Hearts']:
        train_suits.append(Train_suits())
        train_suits[i].name = Suit
        filename = Suit + '.jpg'
        filepath_full = os.path.join(filepath, filename)
        train_suits[i].img = cv2.imread(filepath_full, cv2.IMREAD_GRAYSCALE)
        if train_suits[i].img is None:
            print(f"Warning: Could not load {filepath_full}")
        i = i + 1

    return train_suits


def preprocess_card(img):
    """Preprocess card image for better detection"""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Increase contrast with binary threshold
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)

    return thresh


def flattener(image, pts, w, h):
    """Flattens an image of a card into a top-down 200x300 perspective.
    Returns the flattened, re-sized, grayed image."""
    temp_rect = np.zeros((4, 2), dtype="float32")

    # Find the sum of all coordinates for the points
    # (used to find top-left and bottom-right points)
    s = np.sum(pts, axis=1)

    # Get index of point with smallest sum (top-left)
    tl_idx = np.argmin(s)
    tl = pts[tl_idx][0]

    # Get index of point with largest sum (bottom-right)
    br_idx = np.argmax(s)
    br = pts[br_idx][0]

    # Calculate difference between points
    # (used to find top-right and bottom-left points)
    diff = np.diff(pts, axis=1)

    # Get index of point with smallest difference (top-right)
    tr_idx = np.argmin(diff)
    tr = pts[tr_idx][0]

    # Get index of point with largest difference (bottom-left)
    bl_idx = np.argmax(diff)
    bl = pts[bl_idx][0]

    # Need to create an array listing points in order of
    # [top left, top right, bottom right, bottom left]
    # before doing the perspective transform

    if w <= 0.8 * h:  # If card is vertically oriented
        temp_rect[0] = tl
        temp_rect[1] = tr
        temp_rect[2] = br
        temp_rect[3] = bl

    elif w >= 1.2 * h:  # If card is horizontally oriented
        temp_rect[0] = bl
        temp_rect[1] = tl
        temp_rect[2] = tr
        temp_rect[3] = br

    # If the card is 'diamond' oriented, a different algorithm
    # has to be used to identify which point is top left, top right
    # bottom left, and bottom right.
    elif w > 0.8 * h and w < 1.2 * h:  # If card is diamond oriented
        # If furthest left point is higher than furthest right point,
        # card is tilted to the left.
        if pts[1][0][1] <= pts[3][0][1]:
            # If card is titled to the left, approxPolyDP returns points
            # in this order: top right, top left, bottom left, bottom right
            temp_rect[0] = pts[1][0]  # Top left
            temp_rect[1] = pts[0][0]  # Top right
            temp_rect[2] = pts[3][0]  # Bottom right
            temp_rect[3] = pts[2][0]  # Bottom left

        # If furthest left point is lower than furthest right point,
        # card is tilted to the right
        else:
            # If card is titled to the right, approxPolyDP returns points
            # in this order: top left, bottom left, bottom right, top right
            temp_rect[0] = pts[0][0]  # Top left
            temp_rect[1] = pts[3][0]  # Top right
            temp_rect[2] = pts[2][0]  # Bottom right
            temp_rect[3] = pts[1][0]  # Bottom left

    maxWidth = 200
    maxHeight = 300

    # Create destination array, calculate perspective transform matrix,
    # and warp card image
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], np.float32)
    M = cv2.getPerspectiveTransform(temp_rect, dst)
    warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    warp = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)

    return warp


def find_card_contours(img):
    """Find contours of cards in the image"""
    # Pre-process image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Find background level for adaptive thresholding
    img_h, img_w = gray.shape[:2]
    bkg_level = gray[int(img_h / 100)][int(img_w / 2)]
    thresh_level = bkg_level + BKG_THRESH

    # Apply binary threshold
    _, thresh = cv2.threshold(blur, thresh_level, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    # Initialize empty sorted contour and hierarchy lists
    valid_card_contours = []

    # Determine which of the contours are cards - Find 4 corner points
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.01 * peri, True)

        # If contour has 4 corners (like a card), add it to valid contours
        if len(approx) == 4:
            valid_card_contours.append(approx)

    return valid_card_contours


def extract_card_info_template(img, train_ranks, train_suits, filename_base):
    """Extract card info using template matching based on the GitHub example"""
    # Find contours of cards in the image
    card_contours = find_card_contours(img.copy())

    if not card_contours:
        print(f"No valid card contour found for {filename_base}")
        # Save original for debugging
        cv2.imwrite(f"{filename_base}_original.png", img)
        return "??"

    # Since we're processing ROIs, we're focusing on a single card,
    # so take the largest contour by area
    card_contours.sort(key=cv2.contourArea, reverse=True)
    card_contour = card_contours[0]

    # Create a Query_card object for this contour
    qCard = Query_card()
    qCard.contour = card_contour

    # Get width and height of card
    x, y, w, h = cv2.boundingRect(card_contour)
    qCard.width, qCard.height = w, h

    # Find center of card by taking x and y average of the four corners
    qCard.corner_pts = card_contour
    average = np.sum(card_contour, axis=0) / len(card_contour)
    qCard.center = [int(average[0][0]), int(average[0][1])]

    # Warp card into 200x300 flattened image using perspective transform
    qCard.warp = flattener(img, card_contour, w, h)

    # Save warped card image
    cv2.imwrite(f"{filename_base}_warped.png", qCard.warp)

    # Grab corner of warped card image and do a 4x zoom
    try:
        Qcorner = qCard.warp[0:CORNER_HEIGHT, 0:CORNER_WIDTH]
        Qcorner_zoom = cv2.resize(Qcorner, (0, 0), fx=4, fy=4)

        # Sample known white pixel intensity to determine good threshold level
        white_level = Qcorner_zoom[15, int((CORNER_WIDTH * 4) / 2)]
        thresh_level = white_level - CARD_THRESH
        if thresh_level <= 0:
            thresh_level = 1

        # Threshold the zoomed corner
        _, query_thresh = cv2.threshold(Qcorner_zoom, thresh_level, 255, cv2.THRESH_BINARY_INV)

        # Split corner into top and bottom (top shows rank, bottom shows suit)
        Qrank = query_thresh[20:185, 0:128]
        Qsuit = query_thresh[186:336, 0:128]

        # Find rank contour and bounding rectangle, isolate and find largest contour
        rank_cnts, hier = cv2.findContours(Qrank, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]

        if len(rank_cnts) != 0:
            rank_cnts = sorted(rank_cnts, key=cv2.contourArea, reverse=True)

            # Find bounding rectangle for largest contour, use it to resize query rank image
            x1, y1, w1, h1 = cv2.boundingRect(rank_cnts[0])
            Qrank_roi = Qrank[y1:y1 + h1, x1:x1 + w1]
            Qrank_sized = cv2.resize(Qrank_roi, (RANK_WIDTH, RANK_HEIGHT))
            qCard.rank_img = Qrank_sized

        # Same process with suit
        suit_cnts, hier = cv2.findContours(Qsuit, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]

        if len(suit_cnts) != 0:
            suit_cnts = sorted(suit_cnts, key=cv2.contourArea, reverse=True)

            # Find bounding rectangle for largest contour, use it to resize query suit image
            x2, y2, w2, h2 = cv2.boundingRect(suit_cnts[0])
            Qsuit_roi = Qsuit[y2:y2 + h2, x2:x2 + w2]
            Qsuit_sized = cv2.resize(Qsuit_roi, (SUIT_WIDTH, SUIT_HEIGHT))
            qCard.suit_img = Qsuit_sized

        # Save images of isolated rank and suit
        cv2.imwrite(f"{filename_base}_rank.png", qCard.rank_img)
        cv2.imwrite(f"{filename_base}_suit.png", qCard.suit_img)

        # Match card with train images
        best_rank_match_diff = 10000
        best_suit_match_diff = 10000
        best_rank_match_name = "Unknown"
        best_suit_match_name = "Unknown"

        # If no contours were found in query card, skip matching
        if (len(qCard.rank_img) != 0) and (len(qCard.suit_img) != 0):
            # Match rank
            for Trank in train_ranks:
                if Trank.img is not None and len(Trank.img) > 0:
                    diff_img = cv2.absdiff(qCard.rank_img, Trank.img)
                    rank_diff = int(np.sum(diff_img) / 255)

                    if rank_diff < best_rank_match_diff:
                        best_rank_diff_img = diff_img
                        best_rank_match_diff = rank_diff
                        best_rank_match_name = Trank.name

            # Match suit
            for Tsuit in train_suits:
                if Tsuit.img is not None and len(Tsuit.img) > 0:
                    diff_img = cv2.absdiff(qCard.suit_img, Tsuit.img)
                    suit_diff = int(np.sum(diff_img) / 255)

                    if suit_diff < best_suit_match_diff:
                        best_suit_diff_img = diff_img
                        best_suit_match_diff = suit_diff
                        best_suit_match_name = Tsuit.name

            # Set match results
            if best_rank_match_diff < RANK_DIFF_MAX:
                qCard.best_rank_match = best_rank_match_name
                qCard.rank_diff = best_rank_match_diff

            if best_suit_match_diff < SUIT_DIFF_MAX:
                qCard.best_suit_match = best_suit_match_name
                qCard.suit_diff = best_suit_match_diff

            # Create short name (e.g., "Ah" for Ace of hearts)
            rank_to_short = {
                'Ace': 'A', 'Two': '2', 'Three': '3', 'Four': '4', 'Five': '5',
                'Six': '6', 'Seven': '7', 'Eight': '8', 'Nine': '9', 'Ten': 'T',
                'Jack': 'J', 'Queen': 'Q', 'King': 'K', 'Unknown': '?'
            }

            suit_to_short = {
                'Spades': 's', 'Diamonds': 'd', 'Clubs': 'c', 'Hearts': 'h', 'Unknown': '?'
            }

            short_name = f"{rank_to_short.get(qCard.best_rank_match, '?')}{suit_to_short.get(qCard.best_suit_match, '?')}"

            # Print results
            print(f"Card: {qCard.best_rank_match} of {qCard.best_suit_match}")
            print(f"Short name: {short_name}")
            print(f"Rank diff: {qCard.rank_diff}, Suit diff: {qCard.suit_diff}")

            return short_name

    except Exception as e:
        print(f"Error processing card: {e}")

    return "??"


def extract_card_info_ocr(img, filename_base):
    """Extract card rank and suit using OCR (fallback)"""
    # Save original
    cv2.imwrite(f"{filename_base}.png", img)

    # Preprocess
    processed = preprocess_card(img)
    cv2.imwrite(f"{filename_base}_processed.png", processed)

    # Try to recognize card
    # Set specific config for cards - allowing only valid chars
    config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=AKQJT98765432csdh'

    # Run OCR
    text = pytesseract.image_to_string(processed, config=config).strip().replace('\n', '')

    # Parse the result to find rank and suit
    rank_pattern = r'[AKQJT98765432]'
    suit_pattern = r'[csdh]'

    rank_match = re.search(rank_pattern, text)
    suit_match = re.search(suit_pattern, text)

    rank = rank_match.group(0) if rank_match else '?'
    suit = suit_match.group(0) if suit_match else '?'

    return f"{rank}{suit}"


def run_text_ocr(img, roi_type):
    """Run OCR on text regions"""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Choose config based on type
    if roi_type in ["stack", "bet", "pot"]:
        # For money amounts
        config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789,.$ :'
    else:
        # For names and status
        config = r'--oem 3 --psm 7'

    # Run OCR
    text = pytesseract.image_to_string(gray, config=config).strip()
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# --- Main Execution ---
def main():
    # Check if Card_Imgs directory exists
    if not os.path.exists(CARD_TRAIN_DIR):
        print(f"Warning: Card training directory {CARD_TRAIN_DIR} not found. Template matching may not work.")
        use_template_matching = False
    else:
        use_template_matching = True
        # Load the train rank and suit images
        train_ranks = load_ranks(CARD_TRAIN_DIR)
        train_suits = load_suits(CARD_TRAIN_DIR)
        print("Loaded card training data")

    # Load the image
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print("Error: Could not load image")
        exit()

    print(f"Image loaded. Dimensions: {img.shape[:2]}")

    # Store results
    ocr_results = {
        "players": {},
        "community_cards": [],
        "pot": None
    }

    # Process Player ROIs
    for player_id, rois in PLAYER_ROIS.items():
        print(f"\nProcessing Player: {player_id}")
        ocr_results["players"][player_id] = {}

        for roi_type, roi_data in rois.items():
            if roi_type == "cards":
                ocr_results["players"][player_id]["cards"] = []
                for i, roi in enumerate(roi_data):
                    if roi and len(roi) == 4:
                        x, y, w, h = roi
                        if x >= 0 and y >= 0 and w > 0 and h > 0 and x + w <= img.shape[1] and y + h <= img.shape[0]:
                            roi_img = img[y:y + h, x:x + w]

                            # Process card
                            filename_base = os.path.join(OUTPUT_DIR, f"{player_id}_card_{i + 1}")

                            # Use template matching if available
                            if use_template_matching:
                                card_text = extract_card_info_template(roi_img, train_ranks, train_suits, filename_base)
                            else:
                                card_text = extract_card_info_ocr(roi_img, filename_base)

                            ocr_results["players"][player_id]["cards"].append(card_text)
                            print(f"  Card {i + 1}: {card_text}")
                        else:
                            print(f"  Warning: Invalid ROI for {player_id}_card_{i + 1}")
            elif roi_data and len(roi_data) == 4:
                x, y, w, h = roi_data
                if x >= 0 and y >= 0 and w > 0 and h > 0 and x + w <= img.shape[1] and y + h <= img.shape[0]:
                    roi_img = img[y:y + h, x:x + w]
                    filename = os.path.join(OUTPUT_DIR, f"{player_id}_{roi_type}.png")
                    cv2.imwrite(filename, roi_img)

                    # Process text
                    text = run_text_ocr(roi_img, roi_type)
                    ocr_results["players"][player_id][roi_type] = text
                    print(f"  {roi_type.capitalize()}: {text}")
                else:
                    print(f"  Warning: Invalid ROI for {player_id}_{roi_type}")

    # Process Community Cards
    print("\nProcessing Community Cards:")
    for i, roi in enumerate(COMMUNITY_CARD_ROIS):
        if roi and len(roi) == 4:
            x, y, w, h = roi
            if x >= 0 and y >= 0 and w > 0 and h > 0 and x + w <= img.shape[1] and y + h <= img.shape[0]:
                roi_img = img[y:y + h, x:x + w]

                # Process card
                filename_base = os.path.join(OUTPUT_DIR, f"Community_{i + 1}")

                # Use template matching if available
                if use_template_matching:
                    card_text = extract_card_info_template(roi_img, train_ranks, train_suits, filename_base)
                else:
                    card_text = extract_card_info_ocr(roi_img, filename_base)

                ocr_results["community_cards"].append(card_text)
                print(f"  Card {i + 1}: {card_text}")
            else:
                print(f"  Warning: Invalid ROI for Community Card {i + 1}")

    # Process Pot
    print("\nProcessing Pot:")
    if POT_ROI and len(POT_ROI) == 4:
        x, y, w, h = POT_ROI
        if x >= 0 and y >= 0 and w > 0 and h > 0 and x + w <= img.shape[1] and y + h <= img.shape[0]:
            roi_img = img[y:y + h, x:x + w]
            filename = os.path.join(OUTPUT_DIR, "Pot.png")
            cv2.imwrite(filename, roi_img)

            # Process text
            text = run_text_ocr(roi_img, "pot")
            ocr_results["pot"] = text
            print(f"  Pot: {text}")
        else:
            print(f"  Warning: Invalid ROI for Pot")

    # Print summary
    print("\n=== OCR RESULTS SUMMARY ===")
    print(f"Pot: {ocr_results['pot']}")

    print("\nCommunity Cards:")
    cards_str = ", ".join(ocr_results['community_cards'])
    print(f"  {cards_str}")

    print("\nPlayers Information:")
    for player_id, data in ocr_results["players"].items():
        print(f"\n{player_id}:")
        for key, value in data.items():
            if key == "cards":
                cards_str = ", ".join(value)
                print(f"  Cards: {cards_str}")
            else:
                print(f"  {key.capitalize()}: {value}")

    print("\nFinished processing. Check the images in the output folder.")


if __name__ == "__main__":
    main()
