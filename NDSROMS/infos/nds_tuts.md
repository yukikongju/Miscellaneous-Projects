From ChatGPT

# how to use ndstool on .nds file

You can use the ndstool utility to manipulate Nintendo DS ROMs (NDS files) from the command line. Here are the general steps you can follow to use ndstool:

0. Download and install ndstool. You can find the latest version of ndstool on the devkitPro website.
1. Open a command prompt or terminal window and navigate to the directory where the NDS file is located.
2. To view information about the NDS file, use the following command: `ndstool -i input.nds`
This will display information about the NDS file, such as the title, game code, and file size.
3. To extract the contents of the NDS file, use the following command: `ndstool -x input.nds -9 output.nds`
This will extract the contents of the NDS file into an uncompressed output.nds file.
4. To create a new NDS file, use the following command: `ndstool -c output.nds -9 -p "GAME TITLE" -t "GAME CODE" -d "GAME DESCRIPTION" input.nds`

5. Replace "GAME TITLE", "GAME CODE", and "GAME DESCRIPTION" with the appropriate values for your game. This will create a new NDS file called output.nds with the specified title, code, and description.

6. To replace a file within an existing NDS file, use the following command: `ndstool -x input.nds -9 -r file_to_replace -k replacement_file.nds -c output.nds`

Replace "file_to_replace" with the name of the file you want to replace in the input.nds file, and "replacement_file.nds" with the file you want to replace it with. This will extract the contents of the input.nds file, replace the specified file with the replacement file, and create a new output.nds file.



