# Use an official Node.js image.
FROM node:14

# Set the working directory in the container
WORKDIR /app

# Copy package.json (and package-lock.json if available) and install dependencies.
COPY package.json package-lock.json* ./
RUN npm install

# Copy your application code.
COPY . .

# Expose port 3006 (the port your Node.js server listens on)
EXPOSE 3006

# Define the command to run your Node.js server.
CMD ["node", "server.js"]
