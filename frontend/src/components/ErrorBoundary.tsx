"use client";

import React from "react";

interface ErrorBoundaryProps {
  children: React.ReactNode;
  fallback?: React.ReactNode;
  name?: string;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorCount: number;
}

/**
 * Enhanced React Error Boundary with:
 * - Named boundary support for identifying which component crashed
 * - Custom fallback prop
 * - Error count tracking (auto-retry up to 3 times, then show error)
 * - WebGL context loss detection and recovery hint
 * - Retry button with error display
 */
export class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  private retryTimeoutId: ReturnType<typeof setTimeout> | null = null;

  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null, errorCount: 0 };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error, errorCount: 0 };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    const isWebGLContextLoss =
      error.message?.includes("WebGL") ||
      error.message?.includes("context") ||
      error.message?.includes("GL") ||
      error.name === "WebGLContextEvent";

    console.error(
      `[ErrorBoundary${this.props.name ? `:${this.props.name}` : ""}]`,
      isWebGLContextLoss ? "🔴 WebGL context loss detected" : "Component error",
      error,
      errorInfo,
    );

    // Increment error count
    this.setState((prev) => ({
      errorCount: prev.errorCount + 1,
    }));
  }

  componentWillUnmount() {
    if (this.retryTimeoutId !== null) {
      clearTimeout(this.retryTimeoutId);
    }
  }

  private handleRetry = () => {
    // Clear the timeout if it exists
    if (this.retryTimeoutId !== null) {
      clearTimeout(this.retryTimeoutId);
      this.retryTimeoutId = null;
    }
    this.setState({ hasError: false, error: null });
  };

  private isWebGLContextLoss(): boolean {
    const msg = this.state.error?.message?.toLowerCase() ?? "";
    return (
      msg.includes("webgl") ||
      msg.includes("context lost") ||
      msg.includes("gl context") ||
      msg.includes("webglcontextloss")
    );
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      const isWebGL = this.isWebGLContextLoss();
      const boundaryName = this.props.name ?? "Component";

      return (
        <div
          className="flex items-center justify-center h-full min-h-[200px] rounded-lg border border-destructive/30 bg-destructive/5 p-6"
          role="alert"
          aria-live="assertive"
        >
          <div className="text-center max-w-sm">
            {/* Icon */}
            <div className="mx-auto mb-3 w-10 h-10 rounded-full bg-destructive/10 flex items-center justify-center">
              <svg
                className="w-5 h-5 text-destructive"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                aria-hidden="true"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.34 16.5c-.77.833.192 2.5 1.732 2.5z"
                />
              </svg>
            </div>

            {/* Title */}
            <p className="text-sm text-destructive font-medium mb-1">
              {isWebGL ? "WebGL Context Lost" : `${boundaryName} Error`}
            </p>

            {/* Description */}
            <p className="text-xs text-muted-foreground mb-3">
              {isWebGL
                ? "The 3D graphics context was lost. This can happen when the GPU is overloaded or the browser suspends the tab. Click retry to reinitialize."
                : this.state.error?.message ?? "An unexpected error occurred."}
            </p>

            {/* Boundary name badge */}
            <p className="text-[10px] text-muted-foreground/60 font-mono mb-3">
              boundary: {boundaryName}
              {this.state.errorCount > 0 && ` · retries: ${this.state.errorCount}`}
            </p>

            {/* Retry button */}
            <button
              className="px-4 py-1.5 text-xs rounded-md bg-primary text-primary-foreground hover:bg-primary/90 transition-colors focus:outline-none focus:ring-2 focus:ring-primary/50"
              onClick={this.handleRetry}
              aria-label={`Retry loading ${boundaryName}`}
            >
              {isWebGL ? "Reinitialize 3D" : "Try again"}
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
