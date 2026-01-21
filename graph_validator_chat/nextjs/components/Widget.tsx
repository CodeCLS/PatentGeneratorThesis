'use client';

import { useState } from 'react';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
// Assuming styles are managed by Tailwind CSS now, if not, adjust or remove.
// import styles from './Widget.module.css';

interface WidgetProps {
  type: string;
  data: any;
  onAnswer?: (answer: string) => void;
}

export default function Widget({ type, data, onAnswer }: WidgetProps) {
  const [showAll, setShowAll] = useState(false);

  switch (type) {
    case 'edges_widget':
      const edges = data.triples || [];
      const showCount = 5;
      const displayedEdges = showAll ? edges : edges.slice(0, showCount);
      
      return (
        <div className="rounded-md border">
          <h4 className="p-4 text-lg font-semibold">Triples</h4>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-[50px]">Index</TableHead>
                <TableHead>Head</TableHead>
                <TableHead>Relation</TableHead>
                <TableHead>Tail</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {displayedEdges.map((edge: any, idx: number) => (
                <TableRow key={idx}>
                  <TableCell className="font-medium">
                    {edge.index !== undefined ? edge.index : idx}
                  </TableCell>
                  <TableCell>{edge.head || ''}</TableCell>
                  <TableCell>{edge.relation || ''}</TableCell>
                  <TableCell>{edge.tail || ''}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
          {edges.length > showCount && !showAll && (
            <button className="mt-2 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600" onClick={() => setShowAll(true)}>
              Show More ({edges.length - showCount} remaining)
            </button>
          )}
        </div>
      );

    default:
      return (
        <div className="p-4 border rounded-md">
          <p>Widget: {type}</p>
        </div>
      );
  }
}
