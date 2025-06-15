import React from "react";
import { Alert, AlertIcon, Box, Text, Button } from "@chakra-ui/react";

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { 
      hasError: false, 
      error: null, 
      errorInfo: null,
      errorId: Date.now()
    };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
    
    this.setState({
      error: error,
      errorInfo: errorInfo,
      errorId: Date.now()
    });

    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }
  }

  handleRetry = () => {
    this.setState({ 
      hasError: false, 
      error: null, 
      errorInfo: null,
      errorId: Date.now()
    });
  };

  render() {
    if (this.state.hasError) {
      return (
        <Alert status="error" borderRadius="md" flexDirection="column" alignItems="flex-start">
          <Box display="flex" alignItems="center" mb={2}>
            <AlertIcon />
            <Text fontWeight="bold">Something went wrong</Text>
          </Box>
          
          <Text mb={4} color="gray.600">
            An unexpected error occurred. Please try again or contact support if the problem persists.
          </Text>
          
          {process.env.NODE_ENV === 'development' && (
            <Box
              as="details"
              bg="gray.50"
              p={3}
              borderRadius="md"
              fontSize="sm"
              fontFamily="mono"
              mb={4}
              width="100%"
            >
              <summary>Error details (development mode)</summary>
              <Text mt={2} whiteSpace="pre-wrap">
                {this.state.error && this.state.error.toString()}
                {this.state.errorInfo && this.state.errorInfo.componentStack}
              </Text>
            </Box>
          )}
          
          <Button onClick={this.handleRetry} colorScheme="blue" size="sm">
            Try again
          </Button>
        </Alert>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;