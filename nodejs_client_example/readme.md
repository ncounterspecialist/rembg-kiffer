
## Prerequisites

Ensure that you have [Node.js](https://nodejs.org/) installed.

## Installation

In the root directory of this repository, install the required dependencies:

```bash
npm install
```

Running the Example

Create a .env file 

Add BASE_URL = your_base_url of the cloud run function  like 

```bash
BASE_URL=https://xxxxx.a.run.app/api/remove
```

To run the example:

```bash
npm run example
```

You can modify the imageUrl and imageName parameters in the fetchAndSaveImages function, located in index.js. 

These parameters allow you to specify the image source and name format for saved files.

imageUrl: The URL of the image to process.


imageName: The base name used for saving images and masks.


All processed images and masks will be saved automatically in the images directory.