from doctest import master
import tkinter as tk
from tkinter import StringVar, ttk
import os
from pathlib import Path
from PIL import Image, ImageTk
import pandas as pd
from idlelib.tooltip import Hovertip


OLD_LIST_OF_VIEWS = ['SAX-atria', 'SAX', 'OTHER', '2ch', '3ch', '4ch', 'RVOT', 'LVOT', '2ch-RT', 'RVOT-T', 'Excluded']
LIST_OF_VIEWS = ['SAX-atria', 'SAX', 'SAX-other', '2ch', '3ch', '4ch', 'RVOT', 'LVOT', '2ch-RV', 'RVOT-oblique', 'Excluded']

class VSGUI:
    def __init__(self, patient, dst, viewSelector, show_videos, my_logger):
        self.patient = patient
        self.dst = dst
        self.view_predictions = pd.read_csv(viewSelector.csv_path)
        self.img_dict = {}
        self.viewSelector = viewSelector
        self.my_logger = my_logger
        self.sequence_running = None
        self.show_videos = show_videos
        self.create_window()

    def create_window(self):
        self.window = tk.Tk()

        # Get screen size
        if self.window.winfo_screenwidth() > 1920:
            screen_width = 1920 - 100  # Doesn't need to be larger than 1920
        else:
            screen_width = self.window.winfo_screenwidth() - 100  # Leave some margin
            
        self.scaling = screen_width / 1366  # Scale based on 1366x768 width 
        width = int(1366 * self.scaling) # Set window width to 1366 scaled
        height = int(768 * self.scaling) # Set window height to 768 scaled
        self.window.geometry(f"{width}x{height}")

        self.window.resizable(width=True, height=True)  # Allow resizing of the window

        unique_series = self.view_predictions['Series Number'].unique()
        unique_series = sorted(unique_series, key=lambda x: int(x))  # Sort series numbers numerically

        self.num_rows = 6*2 + 2 # Doubled to allow for buttons above each image
        self.num_cols = 8
        self.gridlayout = {}

        for i in range(self.num_rows):
            for j in range(self.num_cols):
                # Only put images on odd rows
                self.gridlayout[i * self.num_cols + j] = (2*i+2, j)

        # Create a grid layout for the window, with num_rows and num_cols
        self.series_mapping = {}

        for i,s in enumerate(unique_series):
            series = int(s)
            self.series_mapping[series] = i

        # Create a grid layout for the window, with num_rows and num_cols
        self.window.title("View Correction GUI")

        self.window.after(1000, lambda: self.window.focus_force())

    # This function saves the corrected predictions to the processing and states directories
    def save_corrections(self):
        self.my_logger.info("----- Saving view selections...")
        # Get selected option from the drop down menu
        for i, dropdown in enumerate(self.list_of_dropdowns):
            selected_view = dropdown.get()
            series = self.series[i]
            self.view_predictions.loc[self.view_predictions['Series Number'] == series, 'Predicted View'] = selected_view

            self.my_logger.info(f"----- Series {series} saved to {selected_view}.")

        # Save the corrected predictions to the CSV file
        self.view_predictions.to_csv(self.viewSelector.csv_path, index=False)

        # Add text confirmation to the header
        lbl_confirmation = tk.Label(master=self.window, text="View selections saved successfully!", fg="green", font=('Arial', 12))
        lbl_confirmation.grid(row=0, column=5, sticky=tk.W + tk.E, columnspan=2)
        self.my_logger.success("----- View selections saved successfully. Close the window to continue.")

    def display_full_sequence(self, event, idx):
        global loop

        # if self.sequence_running == None:
        #     self.sequence_running = [idx]
        # else:
        #     if idx not in self.sequence_running:
        #         self.stop_full_sequence(event, self.sequence_running[0])
        #         self.sequence_running = [idx]
            
        series = list(self.img_dict.keys())[idx]

        # Replace the previous image with the next image in the sequence
        try:
            next_image = self.img_dict[series][self.counters[idx]]
            self.counters[idx] += 1
        except IndexError: # Otherwise reset to first image
            self.counters[idx] = 0
            next_image = self.img_dict[series][self.counters[idx]]
            self.counters[idx] += 1
       
        # Update label
        self.list_of_images[idx].configure(image=next_image)

        loop = self.window.after(50, lambda event=event, idx=idx: self.display_full_sequence(event,idx))  # Call this function again to display the next frame (therefore making a video)

    def stop_full_sequence(self, event, idx):
        print("stopping full sequence for idx ")

        series = list(self.img_dict.keys())[idx]
        
        self.window.after_cancel(loop)
        self.counter = 0
        # Reset to first image
        first_image = self.img_dict[series][0]
        self.list_of_images[idx].configure(image=first_image)

        self.sequence_running = None

    def correct_views_gui(self):
        # Initialise save button at the top
        btn_optn_confirm = tk.Button(master=self.window, text="Press to save view selections", command=self.save_corrections, borderwidth=5, highlightthickness=5, relief=tk.RAISED, font=('Arial', 18, 'bold'))
        btn_optn_confirm.grid(row=0, column=3, columnspan=2, sticky=tk.W + tk.E)

        # Display the case information at the top
        case_info = tk.Label(master=self.window, text=f"Case: {self.patient}", fg="black", font=('Arial', 16))
        case_info.grid(row=0, column=0, columnspan=2, sticky=tk.W + tk.E)

        # Get directory for png images
        unsorted_img_directory = Path(self.dst, 'view-classification', 'unsorted')

        # Get list of all pngs
        all_imgs = os.listdir(unsorted_img_directory)
        # Format as full paths
        all_imgs = [os.path.join(unsorted_img_directory, i) for i in all_imgs if i.endswith('.png')]
        self.img_dict = {}

        stringvars = []
        confidences = []
        descriptions = []
        locations = []
        frames = []
        self.series = []

        for i in all_imgs:
            decomp_fpath = list(os.path.basename(i).split('_'))
            series = int(decomp_fpath[0])  # Get the series number from the filename
            frame = int(decomp_fpath[1].replace('.png',''))  # Get the frame number from the filename

            if series not in self.img_dict:
                self.img_dict[series] = [(i, frame)]
            else:
                self.img_dict[series].append((i, frame))

        # Set base image size
        base_image_size = 150  # Base size for images in pixels (width=height)
        if len(self.img_dict.keys()) > 24 and len(self.img_dict.keys()) <= 48:
            base_image_size = 100
        elif len(self.img_dict.keys()) > 48:
            base_image_size = 75

        for series in self.img_dict.keys():
            # Get view classification
            vp = self.view_predictions[self.view_predictions['Series Number'] == series]
            view = vp['Predicted View'].values[0]

            stringvars.append(StringVar(master=self.window, value=view))  # Create a StringVar for each image view class
            self.series.append(series)

            confidence = vp['Confidence'].values[0]
            confidences.append(confidence)

            description = vp['Series Description'].values[0]
            descriptions.append(description)

            location = vp['Slice Location'].values[0]
            locations.append(location)

            frames_per_slice = vp['Frames Per Slice'].values[0]
            frames.append(frames_per_slice)

            # Sort images by frame
            sorted_images = sorted(self.img_dict[series], key=lambda x: x[1])  # Sort by frame number
            self.img_dict[series] = sorted_images

            # Load all images and replace paths with loaded images
            loaded_images = []
            for img_path in self.img_dict[series]:
                image = Image.open(img_path[0])
                image = image.resize((int(base_image_size*self.scaling), int(image.size[1] * base_image_size * self.scaling / image.size[0]))) # Scale from width, height of base image size based on screen width
                image_tk = ImageTk.PhotoImage(image)
                loaded_images.append(image_tk)

            self.img_dict[series] = loaded_images

        self.list_of_images = []
        self.list_of_dropdowns = []
        self.counters = []

        # For each file, convert it to a PIL image and display it in a Tkinter widget
        for i, series in enumerate(self.img_dict.keys()):
            images = self.img_dict[series]

            # Display the first frame of each series
            image_tk = images[0]

            mapped_series = self.series_mapping[series]
            
            confidence = confidences[i]
            if confidence < 0.66:
                color = 'orange'
            else:
                color = 'limegreen'

            # Grab view type
            vp = self.view_predictions[self.view_predictions['Series Number'] == series]
            vp = vp['Predicted View'].values[0]

            if vp == "Excluded": # Highlight excluded series in red
                color = 'crimson'

            # Create a label with the image
            lbl_image = tk.Label(master=self.window, image = image_tk, highlightcolor=color, highlightbackground=color,
                                highlightthickness=3, border=3, relief=tk.RAISED)
            lbl_image.anchor(tk.CENTER)
            lbl_image.grid(row=self.gridlayout[mapped_series][0], column=self.gridlayout[mapped_series][1])

            self.list_of_images.append(lbl_image)

            # Add bind to display full sequence on hover
            # lbl_image.bind("<Button-1>", self.display_full_sequence)
            # lbl_image.bind("<Button-3>", lambda event, idx=i: self.stop_full_sequence(event, idx))
            
            # Display full sequence by default
            self.counters.append(0)

            if self.show_videos:
                self.display_full_sequence(None, i) # Comment this line to disable video display

            # Add series number text with outline
            lbl_series = tk.Label(master=self.window, text=f'Series {series}', fg='black', bg='white', border=2, highlightcolor=color, highlightbackground=color, font=('Arial', 10))
            lbl_series.grid(row=self.gridlayout[mapped_series][0], column=self.gridlayout[mapped_series][1], sticky="n")

            if vp == "Excluded" and self.viewSelector.excluded_df is not None:
                # Look for info in excluded df
                excl_row = self.viewSelector.excluded_df[self.viewSelector.excluded_df['Excluded Series'] == series]
                original_view = excl_row['Original View'].values[0]
                reason = excl_row['Reason'].values[0]
                kept = excl_row['Kept Series'].values[0]

                # Add hover text with series number, confidence, description, location, frames, etc
                Hovertip(lbl_image, f'Series: {series}\nOriginal prediction: {vp} ({original_view})\nReason for exclusion: {reason} as series {kept}\nConfidence: {confidences[i]:.2f}\nDescription: {descriptions[i]}\nLocation: {locations[i]:.2f}\nFrames: {frames[i]}', hover_delay=400)

                # Display drop down with list of different views
                # Populate with current view
                self.list_of_dropdowns.append(ttk.Combobox(self.window, values=LIST_OF_VIEWS, textvariable=stringvars[i], state="readonly", font=('Arial', 12), height=11))
                self.list_of_dropdowns[-1].grid(row=self.gridlayout[mapped_series][0]-1, column=self.gridlayout[mapped_series][1])

            else:
                # Add hover text with series number, confidence, description, location, frames, etc
                Hovertip(lbl_image, f'Series: {series}\nOriginal prediction: {vp}\nConfidence: {confidences[i]:.2f}\nDescription: {descriptions[i]}\nLocation: {locations[i]:.2f}\nFrames: {frames[i]}', hover_delay=300)

                self.list_of_dropdowns.append(ttk.Combobox(self.window, values=LIST_OF_VIEWS, textvariable=stringvars[i], state="readonly", font=('Arial', 12), height=11))
                self.list_of_dropdowns[-1].grid(row=self.gridlayout[mapped_series][0]-1, column=self.gridlayout[mapped_series][1])

        # Configure grids
        for i in range(self.num_rows):
            self.window.grid_rowconfigure(i, weight=1)
        for j in range(self.num_cols):
            self.window.grid_columnconfigure(j, weight=1)

        self.window.mainloop()